import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import type { VideoEntry, VideoListResponse, DownloadStatus } from '../types';

const PAGE_SIZE = 50;
const DEBOUNCE_MS = 300;
const POLL_INTERVAL_MS = 2000;

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '24px',
    maxWidth: '1200px',
    margin: '0 auto',
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    color: '#e2e8f0',
    backgroundColor: '#0f172a',
    minHeight: '100vh',
  },
  header: {
    marginBottom: '24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: '16px',
  },
  title: {
    fontSize: '1.75rem',
    fontWeight: 600,
    color: '#f8fafc',
    margin: 0,
  },
  searchInput: {
    padding: '10px 14px',
    fontSize: '0.95rem',
    borderRadius: '8px',
    border: '1px solid #334155',
    backgroundColor: '#1e293b',
    color: '#f8fafc',
    minWidth: '260px',
    outline: 'none',
  },
  searchInputFocus: {
    borderColor: '#64748b',
    boxShadow: '0 0 0 2px rgba(100, 116, 139, 0.2)',
  },
  tableWrapper: {
    overflowX: 'auto',
    borderRadius: '10px',
    border: '1px solid #334155',
    backgroundColor: '#1e293b',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.9rem',
  },
  th: {
    textAlign: 'left',
    padding: '14px 16px',
    fontWeight: 600,
    color: '#94a3b8',
    borderBottom: '1px solid #334155',
    backgroundColor: '#0f172a',
  },
  td: {
    padding: '14px 16px',
    borderBottom: '1px solid #334155',
    color: '#e2e8f0',
  },
  tr: {
    transition: 'background-color 0.15s ease',
  },
  trHover: {
    backgroundColor: '#334155',
  },
  statusReady: {
    color: '#22c55e',
    fontWeight: 500,
  },
  statusNotDownloaded: {
    color: '#64748b',
  },
  statusPending: {
    color: '#fbbf24',
  },
  statusDownloading: {
    color: '#38bdf8',
  },
  statusFailed: {
    color: '#f87171',
  },
  btn: {
    padding: '8px 14px',
    fontSize: '0.875rem',
    fontWeight: 500,
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
  },
  btnPrimary: {
    backgroundColor: '#3b82f6',
    color: '#fff',
  },
  btnSecondary: {
    backgroundColor: '#334155',
    color: '#e2e8f0',
    border: '1px solid #475569',
  },
  btnDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  pagination: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: '20px',
    padding: '16px',
    color: '#94a3b8',
    fontSize: '0.9rem',
  },
  paginationControls: {
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
  },
  loading: {
    textAlign: 'center',
    padding: '48px 24px',
    color: '#64748b',
    fontSize: '1rem',
  },
  error: {
    padding: '16px',
    backgroundColor: '#7f1d1d',
    color: '#fecaca',
    borderRadius: '8px',
    marginBottom: '16px',
  },
};

export default function VideoBrowser() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadStatuses, setDownloadStatuses] = useState<Record<string, DownloadStatus>>({});
  const [searchFocused, setSearchFocused] = useState(false);
  const [hoveredRowId, setHoveredRowId] = useState<string | null>(null);
  const pollIntervalsRef = useRef<Record<string, ReturnType<typeof setInterval>>>({});

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
      setOffset(0);
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [search]);

  // Fetch videos
  const fetchVideos = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        offset: String(offset),
        limit: String(PAGE_SIZE),
      });
      if (debouncedSearch.trim()) {
        params.set('search', debouncedSearch.trim());
      }
      const res = await fetch(`/api/videos?${params.toString()}`);
      if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
      const data: VideoListResponse = await res.json();
      setVideos(data.videos);
      setTotal(data.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load videos');
      setVideos([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [offset, debouncedSearch]);

  useEffect(() => {
    fetchVideos();
  }, [fetchVideos]);

  // Poll status for a video
  const pollStatus = useCallback(
    (videoId: string) => {
      const poll = async () => {
        try {
          const res = await fetch(`/api/videos/${videoId}/status`);
          if (!res.ok) return;
          const data: DownloadStatus = await res.json();
          setDownloadStatuses((prev) => ({ ...prev, [videoId]: data }));

          if (data.status === 'complete' || data.status === 'failed') {
            const id = pollIntervalsRef.current[videoId];
            if (id) {
              clearInterval(id);
              delete pollIntervalsRef.current[videoId];
            }
            if (data.status === 'complete') {
              fetchVideos();
            }
          }
        } catch {
          // ignore poll errors
        }
      };

      poll();
      const id = setInterval(poll, POLL_INTERVAL_MS);
      pollIntervalsRef.current[videoId] = id;
    },
    [fetchVideos]
  );

  // Cleanup poll intervals on unmount
  useEffect(() => {
    return () => {
      Object.values(pollIntervalsRef.current).forEach(clearInterval);
      pollIntervalsRef.current = {};
    };
  }, []);

  const handleDownload = useCallback(
    async (videoId: string) => {
      setDownloadStatuses((prev) => ({ ...prev, [videoId]: { video_id: videoId, status: 'pending', error: null } }));
      try {
        const res = await fetch(`/api/videos/${videoId}/download`, { method: 'POST' });
        if (res.status === 202) {
          pollStatus(videoId);
        } else {
          setDownloadStatuses((prev) => ({
            ...prev,
            [videoId]: { video_id: videoId, status: 'failed', error: `HTTP ${res.status}` },
          }));
        }
      } catch (e) {
        setDownloadStatuses((prev) => ({
          ...prev,
          [videoId]: {
            video_id: videoId,
            status: 'failed',
            error: e instanceof Error ? e.message : 'Request failed',
          },
        }));
      }
    },
    [pollStatus]
  );

  const handleOpen = useCallback(
    (videoId: string) => {
      navigate(`/player/${videoId}`);
    },
    [navigate]
  );

  const getStatusDisplay = (video: VideoEntry): { text: string; styleKey: string } => {
    const ds = downloadStatuses[video.id];
    if (ds) {
      if (ds.status === 'complete') return { text: 'Ready', styleKey: 'statusReady' };
      if (ds.status === 'failed') return { text: `Failed: ${ds.error || 'Unknown'}`, styleKey: 'statusFailed' };
      if (ds.status === 'pending') return { text: 'Pending...', styleKey: 'statusPending' };
      if (ds.status === 'downloading') return { text: 'Downloading...', styleKey: 'statusDownloading' };
    }
    if (video.downloaded) return { text: 'Ready', styleKey: 'statusReady' };
    return { text: 'Not Downloaded', styleKey: 'statusNotDownloaded' };
  };

  const start = offset + 1;
  const end = Math.min(offset + PAGE_SIZE, total);
  const hasPrev = offset > 0;
  const hasNext = offset + PAGE_SIZE < total;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Video Browser</h1>
        <Link
          to="/transcripts"
          style={{
            padding: '8px 14px', fontSize: '0.875rem', fontWeight: 500,
            backgroundColor: '#1e293b', color: '#94a3b8',
            border: '1px solid #334155', borderRadius: '6px',
            textDecoration: 'none', whiteSpace: 'nowrap' as const,
          }}
        >
          Testimony Archive →
        </Link>
        <input
          type="text"
          placeholder="Search videos..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onFocus={() => setSearchFocused(true)}
          onBlur={() => setSearchFocused(false)}
          style={{ ...styles.searchInput, ...(searchFocused ? styles.searchInputFocus : {}) }}
        />
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {loading ? (
        <div style={styles.loading}>Loading videos...</div>
      ) : videos.length === 0 ? (
        <div style={styles.loading}>
          {debouncedSearch ? 'No videos match your search.' : 'No videos in manifest.'}
        </div>
      ) : (
        <>
          <div style={styles.tableWrapper}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Video ID</th>
                  <th style={styles.th}>Subject</th>
                  <th style={styles.th}>Tape</th>
                  <th style={styles.th}>Status</th>
                  <th style={styles.th}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {videos.map((video) => {
                  const { text: statusText, styleKey: statusStyle } = getStatusDisplay(video);
                  const ds = downloadStatuses[video.id];
                  const isDownloading = ds && (ds.status === 'pending' || ds.status === 'downloading');
                  const isDownloaded = video.downloaded || ds?.status === 'complete';
                  return (
                    <tr
                      key={video.id}
                      style={{
                        ...styles.tr,
                        ...(hoveredRowId === video.id ? styles.trHover : {}),
                      }}
                      onMouseEnter={() => setHoveredRowId(video.id)}
                      onMouseLeave={() => setHoveredRowId(null)}
                    >
                      <td style={styles.td}>{video.id}</td>
                      <td style={styles.td}>{video.int_code}</td>
                      <td style={styles.td}>{video.tape}</td>
                      <td style={styles.td}>
                        <span style={styles[statusStyle]}>{statusText}</span>
                      </td>
                      <td style={{ ...styles.td, display: 'flex', gap: 8, alignItems: 'center' }}>
                        {isDownloaded ? (
                          <button
                            style={{ ...styles.btn, ...styles.btnPrimary }}
                            onClick={() => handleOpen(video.id)}
                          >
                            Open
                          </button>
                        ) : (
                          <button
                            style={{
                              ...styles.btn,
                              ...styles.btnSecondary,
                              ...(isDownloading ? styles.btnDisabled : {}),
                            }}
                            onClick={() => !isDownloading && handleDownload(video.id)}
                            disabled={isDownloading}
                          >
                            {isDownloading ? 'Downloading...' : 'Download'}
                          </button>
                        )}
                        <Link
                          to={`/transcript/${video.int_code}`}
                          style={{
                            ...styles.btn, ...styles.btnSecondary,
                            textDecoration: 'none', fontSize: '0.8rem',
                            padding: '6px 10px',
                          }}
                        >
                          Transcript
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {total > 0 && (
            <div style={styles.pagination}>
              <span>
                Showing {start}–{end} of {total}
              </span>
              <div style={styles.paginationControls}>
                <button
                  style={{
                    ...styles.btn,
                    ...styles.btnSecondary,
                    ...(!hasPrev ? styles.btnDisabled : {}),
                  }}
                  onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                  disabled={!hasPrev}
                >
                  Previous
                </button>
                <button
                  style={{
                    ...styles.btn,
                    ...styles.btnSecondary,
                    ...(!hasNext ? styles.btnDisabled : {}),
                  }}
                  onClick={() => setOffset((o) => o + PAGE_SIZE)}
                  disabled={!hasNext}
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
