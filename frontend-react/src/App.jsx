import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { marked } from 'marked';
import { 
  Upload, 
  Globe, 
  Trash2, 
  MessageSquare, 
  BookOpen, 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  Send,
  Plus,
  Search,
  X
} from 'lucide-react';
import './App.css';

const API_BASE = "http://localhost:8000";

function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', content: 'Hello! I am your AI Knowledge Assistant. Upload your documents in the sidebar and ask me anything.' }
  ]);
  const [query, setQuery] = useState('');
  const [urlInput, setUrlInput] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchDocs();
    const interval = setInterval(fetchDocs, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchDocs = async () => {
    try {
      const res = await axios.get(`${API_BASE}/documents`);
      setDocuments(res.data);
    } catch (err) {
      console.error("Fetch Error:", err);
    }
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files.length) return;
    const formData = new FormData();
    for (let file of files) formData.append("files", file);
    await axios.post(`${API_BASE}/upload`, formData);
    fetchDocs();
  };

  const handleAddUrl = async () => {
    if (!urlInput.trim()) return;
    await axios.post(`${API_BASE}/add-url?url=${encodeURIComponent(urlInput)}`);
    setUrlInput('');
    fetchDocs();
  };

  const handleDelete = async (id) => {
    if (!window.confirm("Delete document?")) return;
    await axios.delete(`${API_BASE}/documents/${id}`);
    fetchDocs();
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const selectedIds = Array.from(document.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.id);
    if (selectedIds.length === 0) {
      alert("Please select at least one document.");
      return;
    }

    const userMsgId = Date.now();
    setMessages(prev => [...prev, { id: userMsgId, role: 'user', content: query }]);
    setQuery('');
    setIsLoading(true);

    const botMsgId = userMsgId + 1;
    setMessages(prev => [...prev, { id: botMsgId, role: 'assistant', content: 'Thinking...', isTyping: true }]);

    try {
      const res = await axios.post(`${API_BASE}/query`, {
        query: query,
        selected_doc_ids: selectedIds
      });
      
      setMessages(prev => prev.map(m => 
        m.id === botMsgId ? { ...m, content: res.data.answer, citation: res.data.citation, isTyping: false } : m
      ));
    } catch (err) {
      setMessages(prev => prev.map(m => 
        m.id === botMsgId ? { ...m, content: "Error: Could not get answer.", isTyping: false } : m
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const filteredDocs = documents.filter(doc => 
    doc.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const selectedCount = documents.filter(doc => {
      const cb = document.getElementById(doc.id);
      return cb && cb.checked;
  }).length;

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar glass">
        <div className="p-6">
          <div className="flex items-center gap-2 mb-8">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">ANSR <span className="text-blue-500">RAG</span></h1>
          </div>

          <div className="mb-6">
            <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-4">Knowledge Hub</h2>
            <div className="relative mb-4">
              <Search className="absolute left-3 top-2.5 w-3.5 h-3.5 text-slate-500" />
              <input 
                type="text" 
                placeholder="Filter documents..." 
                className="search-input"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>

            <div className="flex gap-2 mb-4">
              <label className="upload-btn flex-1">
                <Upload className="w-4 h-4" />
                <span>Upload</span>
                <input type="file" multiple className="hidden" onChange={handleUpload} />
              </label>
            </div>

            <div className="flex gap-2">
              <input 
                type="text" 
                placeholder="https://..." 
                className="url-input"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
              />
              <button onClick={handleAddUrl} className="add-btn"><Plus className="w-4 h-4" /></button>
            </div>
          </div>

          <div className="doc-list scroll-hide">
            {filteredDocs.map(doc => (
              <div key={doc.id} className="doc-item group">
                <input 
                  type="checkbox" 
                  id={doc.id} 
                  defaultChecked={doc.status === 'ready'}
                  disabled={doc.status !== 'ready'}
                  className="doc-checkbox" 
                />
                <div className="flex-1 min-w-0">
                  <p className="doc-name">{doc.filename}</p>
                  <div className="flex items-center gap-1.5">
                    {doc.status === 'ready' && <CheckCircle className="w-2.5 h-2.5 text-green-500" />}
                    {doc.status === 'processing' && <Clock className="w-2.5 h-2.5 text-yellow-500" />}
                    {doc.status === 'error' && <AlertCircle className="w-2.5 h-2.5 text-red-500" />}
                    <span className={`status-text ${doc.status}`}>{doc.status}</span>
                  </div>
                </div>
                <button onClick={() => handleDelete(doc.id)} className="delete-btn">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="chat-header glass">
          <div className="flex items-center gap-4">
            <div className="flex -space-x-2">
              <div className="w-8 h-8 rounded-full border-2 border-slate-900 bg-blue-600 flex items-center justify-center text-[10px] font-bold">L3</div>
            </div>
            <div>
              <p className="text-sm font-semibold">Llama 3.3 Engine</p>
              <p className="text-[10px] text-green-500 font-bold uppercase">System Active</p>
            </div>
          </div>
        </header>

        <div className="chat-box scroll-hide">
          {messages.map((m) => (
            <div key={m.id} className={`message-wrapper ${m.role}`}>
              <div className={`message-bubble ${m.role} glass`}>
                <div 
                  className="markdown-body" 
                  dangerouslySetInnerHTML={{ __html: marked.parse(m.content) }} 
                />
                {m.citation && (
                  <div className="citation-card animate-in fade-in">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="source-tag">Source</span>
                      <span className="source-meta">{m.citation.source} | {m.citation.location}</span>
                    </div>
                    <p className="quote">"{m.citation.quote}"</p>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <footer className="chat-input-area">
          <form onSubmit={handleQuery} className="input-container glass">
            <input 
              type="text" 
              placeholder="Ask a question about your documents..." 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" disabled={isLoading} className="send-btn">
              {isLoading ? <Clock className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}

export default App;
