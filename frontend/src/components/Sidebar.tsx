import { UploadCloud, FileText, List, MessageSquare, CheckCircle2, AlertCircle, Info } from "lucide-react";
import { NavLink } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";
import "./Sidebar.css";

interface Status {
  groq: boolean;
  openrouter: boolean;
}

interface SidebarProps {
  status: Status;
  embedded: boolean;
  embedBackend: string;
  theme: "dark" | "light";
  toggleTheme: () => void;
}

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ size?: number }>;
}

export default function Sidebar({ status, embedded, embedBackend: _embedBackend, theme, toggleTheme }: SidebarProps) {
  const navItems: NavItem[] = [
    { name: "Upload Document", path: "/upload", icon: UploadCloud },
    { name: "Summary", path: "/summary", icon: FileText },
    { name: "Clause Analysis", path: "/clauses", icon: List },
    { name: "Ask Questions", path: "/qa", icon: MessageSquare },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">
          Legal Buddy
        </h1>
      </div>

      <nav className="sidebar-nav">
        <p className="nav-label">Navigation</p>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `nav-item ${isActive ? "nav-item-active" : "nav-item-inactive"}`
            }
          >
            <item.icon size={18} />
            <span className="nav-text">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-status">
        <p className="status-label">System Status</p>
        <div className="status-list">
          {/* Vector DB Status */}
          <div className="status-item">
            {embedded ? (
              <CheckCircle2 size={16} className="status-icon status-success" />
            ) : (
              <AlertCircle size={16} className="status-icon status-warning" />
            )}
            <div>
              <p className="status-name">Vector DB</p>
              <p className="status-value">{embedded ? "Ready" : "Not initialized"}</p>
            </div>
          </div>

          {/* Groq API */}
          <div className="status-item">
            {status.groq ? (
              <CheckCircle2 size={16} className="status-icon status-success" />
            ) : (
              <AlertCircle size={16} className="status-icon status-warning" />
            )}
            <div>
              <p className="status-name">Groq API</p>
              <p className="status-value">{status.groq ? "Detected" : "Missing Key"}</p>
            </div>
          </div>

          {/* OpenRouter API */}
          <div className="status-item">
            {status.openrouter ? (
              <CheckCircle2 size={16} className="status-icon status-success" />
            ) : (
              <Info size={16} className="status-icon status-info" />
            )}
            <div>
              <p className="status-name">OpenRouter</p>
              <p className="status-value">{status.openrouter ? "Detected" : "Not Configured"}</p>
            </div>
          </div>
        </div>

        {/* Theme Toggle below System Status list */}
        <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
      </div>
    </aside>
  );
}
