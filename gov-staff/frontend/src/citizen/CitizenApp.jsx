// The citizen assistant, exactly as the citizen app shows it. Its files are
// imported read-only from ../../frontend/src (alias @citizen) and never edited
// here; its /chat and /transcribe calls go to the citizen backend (vite proxy).
import "@citizen/index.css";
import CitizenRoot from "@citizen/App.jsx";

import ChangeUserType from "../components/ChangeUserType";

export default function CitizenApp() {
  return (
    <>
      <ChangeUserType />
      <CitizenRoot />
    </>
  );
}
