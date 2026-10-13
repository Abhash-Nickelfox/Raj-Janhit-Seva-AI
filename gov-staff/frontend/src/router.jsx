// One app, two assistants.
//   /                -> choose user type
//   /citizen         -> citizen assistant (citizen app, read-only; citizen backend)
//   /staff/login, /staff/register, /staff/verify -> staff sign-in pages (public)
//   /staff           -> staff assistant chat (signed-in staff only; staff backend)
import { lazy, Suspense } from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";

import Login from "./auth/Login";
import Register from "./auth/Register";
import RequireAuth from "./auth/RequireAuth";
import VerifyOtp from "./auth/VerifyOtp";
import StaffShell from "./layout/StaffShell";
import ChooseUserType from "./pages/ChooseUserType";
import NotFound from "./pages/NotFound";
import StaffChatPage from "./staff/chat/StaffChatPage";

// loaded only when needed, so the citizen app's global styles stay off staff pages
const CitizenApp = lazy(() => import("./citizen/CitizenApp"));

export const router = createBrowserRouter([
  { path: "/", element: <ChooseUserType /> },
  { path: "/citizen", element: <Suspense fallback={null}><CitizenApp /></Suspense> },
  { path: "/staff", element: <RequireAuth><StaffChatPage /></RequireAuth> },
  {
    path: "/staff",
    element: <StaffShell />,
    children: [
      { path: "login", element: <Login /> },
      { path: "register", element: <Register /> },
      { path: "verify", element: <VerifyOtp /> },
      { path: "*", element: <NotFound /> },
    ],
  },
  // old links
  { path: "/login", element: <Navigate to="/staff/login" replace /> },
  { path: "/register", element: <Navigate to="/staff/register" replace /> },
  { path: "*", element: <NotFound /> },
]);
