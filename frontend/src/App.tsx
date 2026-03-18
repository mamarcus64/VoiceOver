import { BrowserRouter, Routes, Route } from "react-router-dom";
import VideoBrowser from "./components/VideoBrowser";
import PlayerPage from "./components/PlayerPage";
import SmileLogin from "./components/SmileLogin";
import SmileConfig from "./components/SmileConfig";
import SmileAnnotate from "./components/SmileAnnotate";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<VideoBrowser />} />
        <Route path="/player/:videoId" element={<PlayerPage />} />
        <Route path="/smile-login" element={<SmileLogin />} />
        <Route path="/smile-config" element={<SmileConfig />} />
        <Route path="/smile-annotate" element={<SmileAnnotate />} />
      </Routes>
    </BrowserRouter>
  );
}
