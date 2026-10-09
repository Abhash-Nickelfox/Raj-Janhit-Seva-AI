# Persona Chat Flows

Four sample conversations, one per persona, showing what the citizen says and what the bot is expected to say back.

---

## 1. Farmer

**Persona:** Ramlal, 58 years old, small farmer in Rajasthan.

| Turn | Citizen says | Bot should |
|---|---|---|
| 1 | मैं राजस्थान का किसान हूँ, मुझे खेती के लिए कोई सरकारी मदद चाहिए | Ask a simple follow-up question (like age or land size) to figure out which scheme fits |
| 2 | मेरी उम्र 58 साल है और मेरी ज़मीन बहुत छोटी है | Recommend one specific scheme that fits him |
| 3 | Krishi Yantra ke baare mein bataiye | Give details specifically about the Krishi Yantra scheme |
| 4 | ab main Jodhpur shift ho gaya hoon, wahan koi scheme badal jaayegi kya? | Answer based on Jodhpur, his new location |
| 5 | iske alawa koi aur scheme hai kya mere liye? | Suggest any other schemes he may qualify for |

---

## 2. Student Scholarship

**Persona:** Priya, 17 years old, in class 11, OBC category, low family income.

| Turn | Citizen says | Bot should |
|---|---|---|
| 1 | मैं स्टूडेंट हूँ, मुझे पढ़ाई के लिए स्कॉलरशिप चाहिए, हमारी फैमिली इनकम बहुत कम है | Ask basic follow-up questions (like which class she's in) |
| 2 | main abhi 11th mein hoon aur OBC category se hoon | Recommend the scholarship scheme that fits her |
| 3 | kya main Indira Mahila Shakti Udyam Protsahan Yojana ke liye eligible hoon? | Answer specifically about this scheme's eligibility |
| 4 | iske alawa koi aur scholarship scheme hai? | List any other scholarship options available |
| 5 | क्या मुझे हायर एजुकेशन के लिए कोई गवर्नमेंट स्कीम अवेलेबल है | Reply in Hindi |

---

## 3. Job Seeker

**Persona:** Suresh, 12th pass, looking for a government job.

| Turn | Citizen says | Bot should |
|---|---|---|
| 1 | मुझे नौकरी चाहिए | Ask what kind of job or qualification he has |
| 2 | maine 12th pass kiya hai | Ask which location he prefers |
| 3 | Jaipur mein chahiye | Show job openings in Jaipur |
| 4 | पटवारी भर्ती के बारे में बताओ | Give details about the Patwari job opening |
| 5 | aur konsi jobs hai? | Show more job openings |
| 6 | uske liye kaunse documents chahiye? | List the documents needed for that job |

---

## 4. Pension Status

**Persona:** Meera, 65 years old, wants to check if her pension has arrived.

| Turn | Citizen says | Bot should |
|---|---|---|
| 1 | क्या मेरी पेंशन आयी है या नहीं | Ask for her consent to verify identity via Jan Aadhaar |
| 2 | ha | Ask for her 10-digit Jan Aadhaar ID |
| 3 | *(enters a 10-digit ID)* | Send an OTP and ask her to enter it |
| 4 | otp nahi mila, firse bhejo | Resend the OTP |
| 5 | *(enters the correct OTP)* | Tell her whether her pension has been credited |
| 6 | ab mere pati ki pension ke baare mein bhi batao | Start a fresh verification for her husband's status |
