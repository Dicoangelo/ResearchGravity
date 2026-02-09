import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          900: "#0a0e17",
          800: "#111827",
          700: "#1a1f35",
          600: "#2a2f45",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
