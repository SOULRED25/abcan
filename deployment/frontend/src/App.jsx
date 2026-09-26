import React, { useState } from 'react';
import axios from 'axios';

export default function App() {
  const [pdbFile, setPdbFile] = useState(null);
  const [mutation, setMutation] = useState({ chain: 'A', resnum: 45, wild: 'K', mut: 'R' });
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('pdb_file', pdbFile);
    formData.append('chain', mutation.chain);
    formData.append('resnum', mutation.resnum);
    formData.append('wild_aa', mutation.wild);
    formData.append('mut_aa', mutation.mut);

    try {
      const res = await axios.post('http://localhost:8000/predict/single', formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif' }}>
      <h1>abCAN-v2: Antibody-Antigen ΔΔG Prediction</h1>
      
      <div style={{ display: 'flex', gap: '2rem' }}>
        <div style={{ flex: 1, border: '1px solid #ccc', padding: '1rem', borderRadius: '8px' }}>
          <h3>Input Configuration</h3>
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <input type="file" onChange={(e) => setPdbFile(e.target.files[0])} required />
            
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input placeholder="Chain (e.g. A)" value={mutation.chain} onChange={e => setMutation({...mutation, chain: e.target.value})} />
              <input placeholder="ResNum (e.g. 45)" type="number" value={mutation.resnum} onChange={e => setMutation({...mutation, resnum: parseInt(e.target.value)})} />
              <input placeholder="WT (e.g. K)" value={mutation.wild} onChange={e => setMutation({...mutation, wild: e.target.value})} />
              <input placeholder="Mut (e.g. R)" value={mutation.mut} onChange={e => setMutation({...mutation, mut: e.target.value})} />
            </div>
            
            <button type="submit" style={{ padding: '0.5rem', background: '#0066cc', color: 'white', border: 'none', borderRadius: '4px' }}>
              Predict ΔΔG
            </button>
          </form>
        </div>

        <div style={{ flex: 1, border: '1px solid #ccc', padding: '1rem', borderRadius: '8px' }}>
          <h3>Inference Output</h3>
          {result ? (
            <div>
              <h2>{result.ddg_point_estimate.toFixed(2)} kcal/mol</h2>
              <p>Effect: <strong>{result.effect}</strong></p>
              <p>95% CI: [{result.confidence_interval_95[0].toFixed(2)}, {result.confidence_interval_95[1].toFixed(2)}]</p>
              <p>Assay Noise (σ): {result.sigma}</p>
            </div>
          ) : (
            <p>Awaiting prediction...</p>
          )}
        </div>
      </div>
      
      {/* Placeholder for PDB Viewer */}
      <div style={{ marginTop: '2rem', height: '400px', background: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p>Interactive PDB Viewer (Mol* / NGL) mounts here</p>
      </div>
    </div>
  );
}
