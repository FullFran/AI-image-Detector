import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  // basePath: '/AI-image-Detector', // Descomentar si el repo no es username.github.io
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
