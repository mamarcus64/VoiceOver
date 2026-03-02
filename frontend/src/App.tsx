import { BrowserRouter, Routes, Route } from "react-router-dom";
import VideoBrowser from "./components/VideoBrowser";
import PlayerPage from "./components/PlayerPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<VideoBrowser />} />
        <Route path="/player/:videoId" element={<PlayerPage />} />
      </Routes>
    </BrowserRouter>
  );
}
