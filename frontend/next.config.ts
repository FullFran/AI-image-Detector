import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  basePath: '/AI-image-Detector',
  assetPrefix: '/AI-image-Detector/',
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
