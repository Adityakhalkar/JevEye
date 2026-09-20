import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JevEye",
  description:
    "Drop an image, ask a question. A CNN reports what it sees with calibrated confidence; Jev judges what it means.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-neutral-950 text-neutral-200 antialiased">{children}</body>
    </html>
  );
}
