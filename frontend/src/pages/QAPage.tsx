import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Send, Bot, User, FileSearch } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api";
import ReactMarkdown from "react-markdown";
import "./QAPage.css";

interface Chunk {
  source: string;
  content: string;
}

interface Message {
  role: "user" | "assistant" | "system";
  text: string;
  chunks?: Chunk[];
}

interface QAPageProps {
  embedded: boolean;
}

export default function QAPage({ embedded }: QAPageProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [messages, loading]);

  if (!embedded) {
    return (
      <div className="no-document-state">
        <div className="no-document-icon">
          <MessageSquare size={32} />
        </div>
        <h2 className="no-document-title">No Document Embedded</h2>
        <p className="no-document-description">Upload and embed a document first to ask questions.</p>
      </div>
    );
  }

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const data = await askQuestion(userMessage.text);
      const aiMessage: Message = { 
        role: "assistant", 
        text: data.answer,
        chunks: data.chunks || [] 
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err: any) {
      setMessages((prev) => [...prev, { role: "system", text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="qa-page">
      <div className="qa-header">
        <h1 className="qa-title">Q&A Assistant</h1>
        <p className="qa-description">Ask detailed questions about your document.</p>
      </div>

      <div className="chat-container">
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="empty-chat">
              <Bot size={48} className="empty-chat-icon" />
              <p className="empty-chat-title">How can I help you today?</p>
              <p className="empty-chat-description">Ask me anything about the terms, clauses, or risks in the embedded document.</p>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`message ${msg.role === "user" ? "message-user" : "message-assistant"}`}
              >
                <div className={`avatar ${msg.role === "user" ? "avatar-user" : 
                  msg.role === "system" ? "avatar-system" : "avatar-assistant"}`}
                >
                  {msg.role === "user" ? <User size={20} /> : <Bot size={20} />}
                </div>

                <div className={`message-content ${msg.role === "user" ? "message-content-user" : "message-content-assistant"}`}>
                  <div className={`message-bubble ${msg.role === "user" ? "bubble-user" : 
                    msg.role === "system" ? "bubble-system" : "bubble-assistant"}`}
                  >
                    {msg.role === "assistant" ? (
                      <div className="markdown-content">
                        <ReactMarkdown>{msg.text}</ReactMarkdown>
                      </div>
                    ) : (
                      msg.text
                    )}
                  </div>

                  {msg.chunks && msg.chunks.length > 0 && (
                    <div className="sources-container">
                      <details className="sources-details">
                        <summary className="sources-summary">
                          <FileSearch size={16} /> Sources ({msg.chunks.length})
                        </summary>
                        <div className="sources-list">
                          {msg.chunks.map((c, idx) => (
                            <div key={idx} className="source-item">
                              <span className="source-title">{c.source}</span>
                              {c.content}
                            </div>
                          ))}
                        </div>
                      </details>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
            
            {loading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="message message-assistant">
                <div className="avatar avatar-assistant">
                  <Bot size={20} />
                </div>
                <div className="message-content message-content-assistant">
                  <div className="typing-indicator">
                    <span className="typing-dot" style={{ animationDelay: "0ms" }}></span>
                    <span className="typing-dot" style={{ animationDelay: "150ms" }}></span>
                    <span className="typing-dot" style={{ animationDelay: "300ms" }}></span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>

        <div className="input-container">
          <form onSubmit={handleAsk} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything about the contract..."
              disabled={loading}
              className="chat-input"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="send-button"
            >
              <Send size={22} className="send-icon" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
