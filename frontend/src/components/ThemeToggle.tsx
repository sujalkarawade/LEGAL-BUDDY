import { motion } from "framer-motion";
import { Sun, Moon } from "lucide-react";
import "./ThemeToggle.css";

interface ThemeToggleProps {
  theme: "dark" | "light";
  toggleTheme: () => void;
}

export default function ThemeToggle({ theme, toggleTheme }: ThemeToggleProps) {
  return (
    <div className="theme-toggle-container">
      <span className="theme-toggle-label">Theme Mode</span>
      <button
        className={`theme-toggle-btn ${theme}`}
        onClick={toggleTheme}
        aria-label="Toggle Theme"
      >
        <div className="toggle-track">
          <motion.div
            className="toggle-knob"
            layout
            transition={{ type: "spring", stiffness: 500, damping: 30 }}
          >
            {theme === "dark" ? (
              <Moon className="theme-icon moon-icon" size={12} />
            ) : (
              <Sun className="theme-icon sun-icon" size={12} />
            )}
          </motion.div>
        </div>
      </button>
    </div>
  );
}
