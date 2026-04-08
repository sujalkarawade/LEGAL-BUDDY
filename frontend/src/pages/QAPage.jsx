import { motion, AnimatePresence } from "framer-motion";
import { MessageSquare, Send, Bot, User, FileSearch } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api";
import ReactMarkdown from "react-markdown";

export default function QAPage({ embedded }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [messages, loading]);

  if (!embedded) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center text-gray-400 mb-4">
          <MessageSquare size={32} />
        </div>
        <h2 className="text-xl font-medium text-gray-200">No Document Embedded</h2>
        <p className="text-gray-500 text-sm mt-2">Upload and embed a document first to ask questions.</p>
      </div>
    );
  }

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const data = await askQuestion(userMessage.text);
      const aiMessage = { 
        role: "assistant", 
        text: data.answer,
        chunks: data.chunks || [] 
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "system", text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto h-full flex flex-col pt-4">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Q&A Assistant</h1>
        <p className="text-gray-400 text-sm">Ask detailed questions about your document.</p>
      </div>

      <div className="flex-1 glass-panel rounded-2xl flex flex-col overflow-hidden mb-4 relative shadow-2xl">
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center opacity-50">
              <Bot size={48} className="mb-4 text-accent-blue" />
              <p className="text-xl font-medium text-gray-300">How can I help you today?</p>
              <p className="text-base text-gray-500 max-w-md mt-2">Ask me anything about the terms, clauses, or risks in the embedded document.</p>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
              >
                <div className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 mt-1
                  ${msg.role === "user" ? "bg-accent-blue text-white" : 
                    msg.role === "system" ? "bg-red-500/20 text-red-500" : "bg-accent-purple/20 text-accent-purple"}`}
                >
                  {msg.role === "user" ? <User size={20} /> : <Bot size={20} />}
                </div>

                <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`px-6 py-4 rounded-2xl text-lg leading-relaxed
                    ${msg.role === "user" ? "bg-accent-blue text-white rounded-tr-sm shadow-lg shadow-accent-blue/20" : 
                      msg.role === "system" ? "bg-red-500/10 text-red-400 rounded-tl-sm border border-red-500/20" : 
                      "bg-dark-700/80 text-gray-200 border border-white/[0.06] rounded-tl-sm shadow-xl shadow-black/40"}`}
                  >
                    {msg.role === "assistant" ? (
                      <div className="[&_h1]:text-2xl [&_h1]:font-bold [&_h1]:text-white [&_h1]:mb-4 [&_h1]:mt-6 [&_h2]:text-xl [&_h2]:font-semibold [&_h2]:text-white [&_h2]:mb-3 [&_h2]:mt-5 [&_h3]:text-lg [&_h3]:font-semibold [&_h3]:text-gray-100 [&_h3]:mb-2 [&_h3]:mt-4 [&_strong]:text-white [&_strong]:font-bold [&_ul]:list-disc [&_ul]:ml-6 [&_ul]:space-y-2 [&_ul]:mt-3 [&_ol]:list-decimal [&_ol]:ml-6 [&_ol]:space-y-2 [&_ol]:mt-3 [&_li]:text-gray-300 [&_li]:leading-relaxed [&_p]:mb-3 [&_p:last-child]:mb-0">
                        <ReactMarkdown>{msg.text}</ReactMarkdown>
                      </div>
                    ) : (
                      msg.text
                    )}
                  </div>

                  {msg.chunks && msg.chunks.length > 0 && (
                    <div className="w-full mt-2">
                      <details className="group">
                        <summary className="text-sm text-gray-500 cursor-pointer flex items-center gap-1 hover:text-gray-300 transition select-none">
                          <FileSearch size={16} /> Sources ({msg.chunks.length})
                        </summary>
                        <div className="mt-2 space-y-3 pl-3 border-l-2 border-white/10">
                          {msg.chunks.map((c, idx) => (
                <div key={idx} className="bg-dark-950/60 p-4 rounded-lg border border-white/[0.06] text-sm text-gray-400">
                              <span className="text-accent-blue/80 font-medium block mb-1 text-xs uppercase tracking-wider">{c.source}</span>
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
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                <div className="w-10 h-10 rounded-full bg-accent-purple/20 text-accent-purple flex items-center justify-center">
                  <Bot size={20} />
                </div>
                <div className="px-6 py-5 rounded-2xl bg-dark-700/50 border border-white/5 rounded-tl-sm flex gap-1.5 items-center">
                  <span className="w-2.5 h-2.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: "0ms" }}></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: "150ms" }}></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: "300ms" }}></span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>

          <div className="p-6 bg-dark-900/60 border-t border-white/[0.08] backdrop-blur-md">
          <form onSubmit={handleAsk} className="relative flex items-center">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything about the contract..."
              disabled={loading}
              className="w-full bg-dark-950 border border-white/[0.12] rounded-xl pl-5 pr-14 py-4 text-base text-gray-200 placeholder-gray-600 focus:outline-none focus:border-accent-blue/50 focus:ring-1 focus:ring-accent-blue/50 transition-all shadow-inner"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="absolute right-2.5 p-2.5 rounded-lg bg-accent-blue/10 text-accent-blue hover:bg-accent-blue hover:text-white transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={22} className="ml-0.5" />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
