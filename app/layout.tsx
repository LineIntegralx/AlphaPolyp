import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "AlphaPolyp - Polyp Detection & Volume Estimation",
  description: "Advanced AI tool for detecting polyps and estimating their volume from endoscopy images",
  openGraph: {
    images: [
      {
        url: "/images/logo.png",
        width: 400,
        height: 400,
        alt: "AlphaPolyp Logo",
      },
    ],
  },
  icons: {
    icon: "/images/logo.png",
  },
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
