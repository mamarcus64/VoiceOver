import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import VideoBrowser from "./components/VideoBrowser";
import PlayerPage from "./components/PlayerPage";
import SmileLogin from "./components/SmileLogin";
import SmileConfig from "./components/SmileConfig";
import SmileAnnotate from "./components/SmileAnnotate";
import SmileAgreement from "./components/SmileAgreement";

const HIDE_DEV_TABS = import.meta.env.VITE_HIDE_DEV_TABS === "true";

export default function App() {
  if (HIDE_DEV_TABS) {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/smile-login" replace />} />
          <Route path="/smile-login" element={<SmileLogin />} />
          <Route path="/smile-annotate" element={<SmileAnnotate />} />
          <Route path="/agreement" element={<SmileAgreement />} />
          <Route path="/pilot-smile-annotate" element={<SmileAnnotate apiPrefix="pilot-" />} />
          <Route path="/pilot-agreement" element={<SmileAgreement apiPrefix="pilot-" />} />
          <Route path="*" element={<Navigate to="/smile-login" replace />} />
        </Routes>
      </BrowserRouter>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<VideoBrowser />} />
        <Route path="/player/:videoId" element={<PlayerPage />} />
        <Route path="/smile-login" element={<SmileLogin />} />
        <Route path="/smile-config" element={<SmileConfig />} />
        <Route path="/smile-annotate" element={<SmileAnnotate />} />
        <Route path="/agreement" element={<SmileAgreement />} />
        <Route path="/pilot-smile-annotate" element={<SmileAnnotate apiPrefix="pilot-" />} />
        <Route path="/pilot-agreement" element={<SmileAgreement apiPrefix="pilot-" />} />
      </Routes>
    </BrowserRouter>
  );
}
