import { useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import "./App.css";
import Dashboard from "./pages/Dashboard";

export default function App() {
  const navigate = useNavigate();

  return (
    <div className="app-container">
      <Routes>
        <Route path="/" element={<Dashboard />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
