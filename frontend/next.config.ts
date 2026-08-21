import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  rewrites: async () => {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
  allowedDevOrigins: ["localhost:3000", "192.168.0.107:3000"],
} as any;

export default nextConfig;
