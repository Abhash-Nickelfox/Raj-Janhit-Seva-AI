// Small floating link back to "Choose user type".
import { Link } from "react-router-dom";

import { useT } from "../i18n";

export default function ChangeUserType() {
  const { t } = useT();
  return (
    <Link to="/" className="change-type" aria-label={t("choose.change")}>
      ← {t("choose.change")}
    </Link>
  );
}
