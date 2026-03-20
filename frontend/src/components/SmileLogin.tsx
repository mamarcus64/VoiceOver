import { useState, useEffect } from "react";
import type { FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

const STORAGE_KEY = "smile_annotator_name";
const HELP_SEEN_KEY = "smile_help_seen";

const st: Record<string, React.CSSProperties> = {
  page: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    minHeight: "100vh",
    backgroundColor: "#0f172a",
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },
  card: {
    backgroundColor: "#1e293b",
    borderRadius: "12px",
    padding: "40px",
    width: "360px",
    boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
  },
  title: {
    fontSize: "1.5rem",
    fontWeight: 700,
    color: "#f8fafc",
    marginBottom: "8px",
    textAlign: "center" as const,
  },
  subtitle: {
    fontSize: "0.85rem",
    color: "#94a3b8",
    marginBottom: "28px",
    textAlign: "center" as const,
  },
  label: {
    display: "block",
    fontSize: "0.8rem",
    fontWeight: 600,
    color: "#94a3b8",
    marginBottom: "6px",
  },
  input: {
    width: "100%",
    padding: "10px 12px",
    fontSize: "0.95rem",
    border: "1px solid #475569",
    borderRadius: "6px",
    backgroundColor: "#0f172a",
    color: "#e2e8f0",
    marginBottom: "16px",
    boxSizing: "border-box" as const,
    outline: "none",
  },
  button: {
    width: "100%",
    padding: "12px",
    fontSize: "1rem",
    fontWeight: 600,
    border: "none",
    borderRadius: "8px",
    backgroundColor: "#3b82f6",
    color: "#fff",
    cursor: "pointer",
    marginTop: "8px",
  },
  error: {
    color: "#f87171",
    fontSize: "0.85rem",
    textAlign: "center" as const,
    marginTop: "12px",
  },
};

function safeNextPath(raw: string | null): string | null {
  if (!raw || !raw.startsWith("/") || raw.startsWith("//")) return null;
  return raw;
}

export default function SmileLogin() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const dest = safeNextPath(searchParams.get("next")) ?? "/smile-annotate";
      navigate(dest, { replace: true });
    }
  }, [navigate, searchParams]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await fetch("/api/smile-auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), password }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail ?? `Error ${res.status}`);
      }
      const data = await res.json();
      localStorage.setItem(STORAGE_KEY, data.annotator);
      const isFirstLogin = !localStorage.getItem(HELP_SEEN_KEY);
      if (isFirstLogin) localStorage.setItem(HELP_SEEN_KEY, "1");
      const dest = safeNextPath(searchParams.get("next")) ?? "/smile-annotate";
      navigate(dest, { replace: true, state: { showHelp: isFirstLogin && dest === "/smile-annotate" } });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={st.page}>
      <form style={st.card} onSubmit={handleSubmit}>
        <div style={st.title}>Smile Annotation</div>
        <div style={st.subtitle}>Log in to start annotating</div>

        <label style={st.label}>Your Name</label>
        <input
          style={st.input}
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Jordan"
          autoFocus
          required
        />

        <label style={st.label}>Password</label>
        <input
          style={st.input}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter password"
          required
        />

        <button style={st.button} type="submit" disabled={loading}>
          {loading ? "Logging in..." : "Log In"}
        </button>

        {error && <div style={st.error}>{error}</div>}
      </form>
    </div>
  );
}
