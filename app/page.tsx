import Image from "next/image"
import { UploadForm } from "@/components/upload-form"
import { Features } from "@/components/features"

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-900 text-white">
      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 flex flex-col items-center">
        <div className="flex items-center gap-4 mb-8">
          <Image src="/images/logo.png" alt="AlphaPolyp Logo" width={100} height={100} />
          <h1 className="text-4xl md:text-5xl font-bold">AlphaPolyp</h1>
        </div>
        <h2 className="text-xl md:text-2xl text-center max-w-2xl mb-8 text-slate-300">
          Advanced polyp detection and volume estimation using AI
        </h2>

        <UploadForm />
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-16">
        <Features />
      </section>
    </main>
  )
}
