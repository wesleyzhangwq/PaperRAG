/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: { primary: '#FAF9F7', secondary: '#F5F4F0', card: '#FFFFFF' },
        text: { primary: '#1A1A1A', secondary: '#6B6560', tertiary: '#9B9590' },
        accent: { DEFAULT: '#D97706', light: '#FEF3C7' },
        border: '#E8E5E0',
      },
      borderRadius: { card: '12px', sm: '8px' },
      boxShadow: { card: '0 1px 3px rgba(0,0,0,0.04)' },
      fontFamily: {
        sans: ['-apple-system', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
