"""Generic RSA+AES envelope crypto - sign -> encrypt -> wrap, matching the
Jan Aadhaar Integration Document v1.8's encryption model exactly (Section on
encryption: build JSON, sign with private key, AES-encrypt the signed JSON,
RSA-wrap the AES key, send as "ivBase64:encryptedDataBase64:encryptedAESKey").

Used by BOTH sides of the mock - the consumer-side client
(app/integrations/janaadhaar/client.py) and the mock server routes
(app/integrations/janaadhaar/mock_routes.py) - each with its OWN keypair, so the mock
genuinely simulates two distinct parties exchanging signed/encrypted
messages instead of one party signing its own messages with a single key.
Real integration would replace these two generated keypairs with our actual
department keypair and Jan Aadhaar's real public key, exchanged during
onboarding - nothing else about seal()/open_envelope() would change.
"""

import base64
import json
import secrets

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_OAEP = asym_padding.OAEP(mgf=asym_padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
_PSS = asym_padding.PSS(mgf=asym_padding.MGF1(hashes.SHA256()), salt_length=asym_padding.PSS.MAX_LENGTH)


class Keypair:
    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()


# Stand-ins for the two real parties until actual keys are exchanged during
# departmental onboarding. Generated once per process - both the client and
# the mock server import these same two objects, so "our" and "Jan
# Aadhaar's" keys stay consistent across a whole run without a real
# exchange having happened.
CONSUMER_KEYPAIR = Keypair()
JANAADHAAR_KEYPAIR = Keypair()


def seal(inner: dict, wrapper_key: str, sign_key, encrypt_key) -> str:
    """wrapper_key is "data" for a REQUEST payload or "response" for a
    RESPONSE payload, per the doc - the encrypted envelope shape is
    identical either direction, only this outer key name differs."""
    body = json.dumps(inner, sort_keys=True).encode()
    signature = sign_key.sign(body, _PSS, hashes.SHA256())
    signed = json.dumps(
        {wrapper_key: inner, "signature": base64.b64encode(signature).decode()}, sort_keys=True
    ).encode()

    aes_key = AESGCM.generate_key(bit_length=256)
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, signed, None)
    wrapped_key = encrypt_key.encrypt(aes_key, _OAEP)

    return ":".join(base64.b64encode(part).decode() for part in (nonce, ciphertext, wrapped_key))


def open_envelope(sealed: str, wrapper_key: str, decrypt_key, verify_key) -> dict:
    nonce_b64, ciphertext_b64, wrapped_key_b64 = sealed.split(":")
    nonce = base64.b64decode(nonce_b64)
    ciphertext = base64.b64decode(ciphertext_b64)
    wrapped_key = base64.b64decode(wrapped_key_b64)

    aes_key = decrypt_key.decrypt(wrapped_key, _OAEP)
    signed = json.loads(AESGCM(aes_key).decrypt(nonce, ciphertext, None))

    signature = base64.b64decode(signed["signature"])
    inner = signed[wrapper_key]
    body = json.dumps(inner, sort_keys=True).encode()
    verify_key.verify(signature, body, _PSS, hashes.SHA256())  # raises if tampered
    return inner
