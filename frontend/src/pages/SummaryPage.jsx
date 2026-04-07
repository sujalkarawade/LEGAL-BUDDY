import { motion } from "framer-motion";
import { FileText, Loader2, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";

export default function SummaryPage({ summary, handleSummarize, loading, embedded, error }) {
  if (!embedded) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center">
        <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center text-gray-400 mb-4">
          <FileText size={32} />
        </div>
        <h2 className="text-xl font-medium text-gray-200">No Document Embedded</h2>
        <p className="text-gray-500 text-sm mt-2">Upload and embed a document first to generate a summary.</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-7xl mx-auto"
    >
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Document Summary</h1>
          <p className="text-gray-400 text-sm">AI-generated complete summary of the key points.</p>
        </div>
        <button
          onClick={handleSummarize}
          disabled={!!loading}
          className="bg-accent-blue hover:bg-blue-600 text-white px-5 py-2.5 rounded-xl text-sm font-medium transition-all flex items-center gap-2 disabled:opacity-50"
        >
          {loading ? (
            <><Loader2 size={16} className="animate-spin" /> Generating...</>
          ) : (
            <><Sparkles size={16} /> Generate Summary</>
          )}
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          {error}
        </div>
      )}

      {summary ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="glass-panel rounded-2xl p-8"
        >
          <div className="text-lg text-gray-300 leading-9 space-y-4 [&_h1]:text-3xl [&_h1]:font-bold [&_h1]:text-white [&_h1]:mb-4 [&_h1]:mt-6 [&_h2]:text-2xl [&_h2]:font-semibold [&_h2]:text-white [&_h2]:mb-3 [&_h2]:mt-5 [&_h3]:text-xl [&_h3]:font-semibold [&_h3]:text-gray-100 [&_h3]:mb-2 [&_h3]:mt-4 [&_strong]:text-white [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:ml-6 [&_ul]:space-y-2 [&_ol]:list-decimal [&_ol]:ml-6 [&_ol]:space-y-2 [&_li]:text-gray-300 [&_li]:leading-9 [&_p]:mb-4">
            <ReactMarkdown>{summary}</ReactMarkdown>
          </div>
        </motion.div>
      ) : (
        !loading && (
          <div className="glass-panel rounded-2xl p-12 text-center border border-dashed border-white/10 mt-6">
            <Sparkles size={32} className="mx-auto text-gray-500 mb-4" />
            <p className="text-gray-400">Click generate to create an AI summary.</p>
          </div>
        )
      )}
    </motion.div>
  );
}

