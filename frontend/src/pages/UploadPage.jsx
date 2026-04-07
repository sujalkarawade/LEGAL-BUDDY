import { motion } from "framer-motion";
import { UploadCloud, File, Activity, Loader2 } from "lucide-react";

export default function UploadPage({ filename, embedded, loading, error, handleUpload, handleEmbed }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-2xl mx-auto mt-10"
    >
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold mb-2">Upload Legal Document</h1>
        <p className="text-gray-400 text-sm">Upload a PDF contract or legal file to build embeddings and begin analysis.</p>
      </div>

      <div className="glass-panel rounded-2xl p-8 card-hover relative overflow-hidden text-center border border-dashed border-white/20">
        <input
          type="file"
          accept=".pdf"
          id="file-upload"
          className="hidden"
          onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])}
        />
        
        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-4">
          <div className="w-16 h-16 rounded-full bg-accent-blue/10 flex items-center justify-center text-accent-blue">
            <UploadCloud size={32} />
          </div>
          <div>
            <p className="text-lg font-medium text-gray-200">Click to upload or drag and drop</p>
            <p className="text-sm text-gray-500 mt-1">PDF documents only</p>
          </div>
        </label>
      </div>

      {filename && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mt-6 glass-panel rounded-xl p-4 flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/5 rounded-lg">
              <File size={20} className="text-blue-400" />
            </div>
            <div>
              <p className="text-sm font-medium">{filename}</p>
              <p className="text-xs text-gray-400">Ready to embed</p>
            </div>
          </div>
          
          <button
            onClick={handleEmbed}
            disabled={!!loading || embedded}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2
              ${embedded 
                ? 'bg-green-500/20 text-green-400 border border-green-500/20 cursor-default'
                : 'bg-accent-blue hover:bg-blue-600 text-white'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {loading ? (
              <><Loader2 size={16} className="animate-spin" /> Processing</>
            ) : embedded ? (
              "Embedded"
            ) : (
              <><Activity size={16} /> Embed Document</>
            )}
          </button>
        </motion.div>
      )}

      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          {error}
        </motion.div>
      )}
    </motion.div>
  );
}
