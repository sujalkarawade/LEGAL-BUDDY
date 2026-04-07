import { motion } from "framer-motion";
import { UploadCloud, FileText, List, MessageSquare, Database, CheckCircle2, AlertCircle, Info } from "lucide-react";
import { NavLink } from "react-router-dom";

export default function Sidebar({ status, embedded, embedBackend }) {
  const navItems = [
    { name: "Upload Document", path: "/upload", icon: UploadCloud },
    { name: "Summary", path: "/summary", icon: FileText },
    { name: "Clause Analysis", path: "/clauses", icon: List },
    { name: "Ask Questions", path: "/qa", icon: MessageSquare },
  ];

  return (
    <aside className="w-64 flex-shrink-0 bg-dark-800 border-r border-white/[0.06] flex flex-col h-full">
      <div className="p-6 border-b border-white/[0.06]">
        <h1 className="text-xl font-bold bg-gradient-to-r from-accent-blue to-accent-purple bg-clip-text text-transparent flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center text-white">
            <List size={18} />
          </div>
          Legal Buddy
        </h1>
      </div>

      <nav className="flex-1 p-4 space-y-2">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 ml-2">Navigation</p>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
                isActive
                  ? "bg-accent-blue/10 text-accent-blue"
                  : "text-gray-400 hover:bg-white/5 hover:text-gray-200"
              }`
            }
          >
            <item.icon size={18} />
            <span className="font-medium text-sm">{item.name}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-white/[0.06] bg-black/40">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 px-2">Status</p>
        <div className="space-y-3">
          {/* Vector DB Status */}
          <div className="flex items-center gap-3 px-2">
            {embedded ? (
              <CheckCircle2 size={16} className="text-green-400" />
            ) : (
              <AlertCircle size={16} className="text-yellow-400" />
            )}
            <div>
              <p className="text-sm font-medium text-gray-300">Vector DB</p>
              <p className="text-xs text-gray-500">{embedded ? "Ready" : "Not initialized"}</p>
            </div>
          </div>

          {/* Groq API */}
          <div className="flex items-center gap-3 px-2">
            {status.groq ? (
              <CheckCircle2 size={16} className="text-green-400" />
            ) : (
              <AlertCircle size={16} className="text-yellow-400" />
            )}
            <div>
              <p className="text-sm font-medium text-gray-300">Groq API</p>
              <p className="text-xs text-gray-500">{status.groq ? "Detected" : "Missing Key"}</p>
            </div>
          </div>

          {/* OpenRouter API */}
          <div className="flex items-center gap-3 px-2">
            {status.openrouter ? (
              <CheckCircle2 size={16} className="text-green-400" />
            ) : (
              <Info size={16} className="text-blue-400" />
            )}
            <div>
              <p className="text-sm font-medium text-gray-300">OpenRouter</p>
              <p className="text-xs text-gray-500">{status.openrouter ? "Detected" : "Not Configured"}</p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
