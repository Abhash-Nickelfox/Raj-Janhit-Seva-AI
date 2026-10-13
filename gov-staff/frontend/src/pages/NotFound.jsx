import { Link } from "react-router-dom";

import { useT } from "../i18n";

export default function NotFound() {
  const { t } = useT();
  return (
    <div className="welcome">
      <h1>{t("notFound.title")}</h1>
      <Link to="/">{t("notFound.home")}</Link>
    </div>
  );
}
