"use client";

import { useCallback, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ClassificationResult {
  prediction: string;
  probability_real: number;
  probability_fake: number;
  processing_time_ms: number;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [visualResult, setVisualResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setVisualResult(null);
      setError(null);
    }
  }, []);

  const handleClassify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Clasificación JSON
      const response = await fetch(`${API_URL}/classify`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Error en la clasificación");

      const data = await response.json();
      setResult(data);

      // Visualización
      const formData2 = new FormData();
      formData2.append("file", file);
      const vizResponse = await fetch(`${API_URL}/analyze-visual`, {
        method: "POST",
        body: formData2,
      });

      if (vizResponse.ok) {
        const blob = await vizResponse.blob();
        setVisualResult(URL.createObjectURL(blob));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setLoading(false);
    }
  };

  const isPredictionReal = result?.prediction === "REAL";

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-400">
            🔬 AI Image Detector
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Detecta si una imagen fue generada por IA analizando sus patrones de gradiente de luminancia
          </p>
        </header>

        {/* Main Content */}
        <main className="max-w-4xl mx-auto">
          {/* Upload Section */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 mb-8 border border-white/20">
            <div className="flex flex-col items-center">
              <label
                htmlFor="file-upload"
                className="w-full max-w-md h-48 border-2 border-dashed border-cyan-400/50 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-cyan-400 hover:bg-cyan-400/5 transition-all"
              >
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    className="h-full w-full object-contain rounded-xl"
                  />
                ) : (
                  <>
                    <svg className="w-12 h-12 text-cyan-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="text-gray-300">Arrastra o haz clic para subir imagen</span>
                  </>
                )}
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>

              <button
                onClick={handleClassify}
                disabled={!file || loading}
                className="mt-6 px-8 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 text-white font-semibold rounded-full hover:from-cyan-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105"
              >
                {loading ? (
                  <span className="flex items-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analizando...
                  </span>
                ) : (
                  "🔍 Analizar Imagen"
                )}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-500/20 border border-red-500 rounded-xl p-4 mb-8 text-red-300 text-center">
              {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-6 text-center">Resultado del Análisis</h2>
              
              <div className={`text-center p-6 rounded-xl mb-6 ${isPredictionReal ? 'bg-green-500/20 border border-green-500' : 'bg-red-500/20 border border-red-500'}`}>
                <div className={`text-4xl font-bold ${isPredictionReal ? 'text-green-400' : 'text-red-400'}`}>
                  {result.prediction}
                </div>
                <div className="text-gray-300 mt-2">
                  Confianza: {((isPredictionReal ? result.probability_real : result.probability_fake) * 100).toFixed(1)}%
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-green-500/10 rounded-xl p-4 text-center">
                  <div className="text-sm text-gray-400">Probabilidad Real</div>
                  <div className="text-2xl font-bold text-green-400">{(result.probability_real * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-red-500/10 rounded-xl p-4 text-center">
                  <div className="text-sm text-gray-400">Probabilidad IA</div>
                  <div className="text-2xl font-bold text-red-400">{(result.probability_fake * 100).toFixed(1)}%</div>
                </div>
              </div>

              <div className="text-center text-gray-400 text-sm">
                Tiempo de procesamiento: {result.processing_time_ms.toFixed(0)}ms
              </div>

              {/* Visual Analysis */}
              {visualResult && (
                <div className="mt-8">
                  <h3 className="text-xl font-semibold text-white mb-4 text-center">Análisis Visual de Gradientes</h3>
                  <img src={visualResult} alt="Visual Analysis" className="w-full rounded-xl" />
                </div>
              )}
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="text-center mt-12 text-gray-500">
          <p>Basado en análisis de covarianza de gradientes de luminancia (PCA)</p>
        </footer>
      </div>
    </div>
  );
}
