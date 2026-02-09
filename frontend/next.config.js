/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // Extend server timeout for long-running LLM requests
  experimental: {
    proxyTimeout: 300000, // 5 menit
  },
  
  httpAgentOptions: {
    keepAlive: true,
  },
  
  // Proxy rewrites untuk development
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: `${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'}/:path*`,
      },
    ];
  },
  
  // Allow external images if needed
  images: {
    remotePatterns: [],
  },
};

module.exports = nextConfig;
