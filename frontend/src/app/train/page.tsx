"use client";

import Link from "next/link";
import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function TrainPage() {
  const [realFiles, setRealFiles] = useState<File[]>([]);
  const [fakeFiles, setFakeFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRealFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setRealFiles(Array.from(e.target.files));
    }
  };

  const handleFakeFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFakeFiles(Array.from(e.target.files));
    }
  };

  const handleTrain = async () => {
    if (realFiles.length === 0 || fakeFiles.length === 0) {
      setError("Necesitas subir al menos 1 imagen de cada tipo");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      realFiles.forEach((file) => formData.append("real_images", file));
      fakeFiles.forEach((file) => formData.append("fake_images", file));

      const response = await fetch(`${API_URL}/train`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Error en el entrenamiento");
      }

      const data = await response.json();
      setResult({
        success: true,
        message: `Modelo entrenado con ${data.n_real_images} imágenes reales y ${data.n_fake_images} imágenes IA`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <header className="text-center mb-12">
          <Link href="/" className="text-cyan-400 hover:text-cyan-300 mb-4 inline-block">
            ← Volver al Detector
          </Link>
          <h1 className="text-5xl font-bold text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-400">
            🎓 Entrenar el Modelo
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Sube imágenes reales y generadas por IA para entrenar el clasificador
          </p>
        </header>

        {/* Main Content */}
        <main className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            {/* Real Images */}
            <div className="bg-green-500/10 backdrop-blur-lg rounded-2xl p-8 border border-green-500/30">
              <h2 className="text-2xl font-bold text-green-400 mb-4 text-center">📷 Imágenes Reales</h2>
              <p className="text-gray-400 text-sm text-center mb-6">Fotos de cámara, sin editar</p>
              
              <label
                htmlFor="real-upload"
                className="w-full h-32 border-2 border-dashed border-green-400/50 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-green-400 hover:bg-green-400/5 transition-all"
              >
                <svg className="w-10 h-10 text-green-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span className="text-gray-300">Seleccionar imágenes</span>
              </label>
              <input
                id="real-upload"
                type="file"
                accept="image/*"
                multiple
                onChange={handleRealFilesChange}
                className="hidden"
              />
              
              {realFiles.length > 0 && (
                <div className="mt-4 text-green-400 text-center">
                  ✓ {realFiles.length} imagen(es) seleccionada(s)
                </div>
              )}
            </div>

            {/* Fake Images */}
            <div className="bg-red-500/10 backdrop-blur-lg rounded-2xl p-8 border border-red-500/30">
              <h2 className="text-2xl font-bold text-red-400 mb-4 text-center">🤖 Imágenes IA</h2>
              <p className="text-gray-400 text-sm text-center mb-6">Generadas por Midjourney, DALL-E, etc.</p>
              
              <label
                htmlFor="fake-upload"
                className="w-full h-32 border-2 border-dashed border-red-400/50 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-red-400 hover:bg-red-400/5 transition-all"
              >
                <svg className="w-10 h-10 text-red-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span className="text-gray-300">Seleccionar imágenes</span>
              </label>
              <input
                id="fake-upload"
                type="file"
                accept="image/*"
                multiple
                onChange={handleFakeFilesChange}
                className="hidden"
              />
              
              {fakeFiles.length > 0 && (
                <div className="mt-4 text-red-400 text-center">
                  ✓ {fakeFiles.length} imagen(es) seleccionada(s)
                </div>
              )}
            </div>
          </div>

          {/* Train Button */}
          <div className="text-center mb-8">
            <button
              onClick={handleTrain}
              disabled={realFiles.length === 0 || fakeFiles.length === 0 || loading}
              className="px-12 py-4 bg-gradient-to-r from-cyan-500 to-purple-500 text-white text-xl font-semibold rounded-full hover:from-cyan-600 hover:to-purple-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-6 w-6 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Entrenando...
                </span>
              ) : (
                "🚀 Entrenar Modelo"
              )}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-500/20 border border-red-500 rounded-xl p-4 mb-8 text-red-300 text-center">
              {error}
            </div>
          )}

          {/* Success */}
          {result?.success && (
            <div className="bg-green-500/20 border border-green-500 rounded-xl p-6 text-center">
              <div className="text-3xl mb-2">✅</div>
              <div className="text-green-400 text-xl font-semibold">{result.message}</div>
              <Link
                href="/"
                className="mt-4 inline-block px-6 py-2 bg-cyan-500 text-white rounded-full hover:bg-cyan-600 transition-colors"
              >
                Ir a Clasificar →
              </Link>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
