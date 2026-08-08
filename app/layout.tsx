import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'StrideSight — AI Sprint Biomechanics Analysis',
  description:
    'On-device computer vision for frame-by-frame sprint biomechanics: knee drive, hip extension, and arm swing.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-neutral-950`}>{children}</body>
    </html>
  );
}
