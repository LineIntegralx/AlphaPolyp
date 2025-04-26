import { Card } from "@/components/ui/card"
import { Upload, Search, BarChart3, Zap } from "lucide-react"

export function Features() {
  const features = [
    {
      icon: <Upload className="h-8 w-8 text-[#d15642]" />,
      title: "Easy Upload",
      description: "Simply drag and drop your endoscopy images for instant analysis",
    },
    {
      icon: <Search className="h-8 w-8 text-[#d15642]" />,
      title: "Precise Detection",
      description: "Advanced AI algorithms accurately identify and locate polyps",
    },
    {
      icon: <BarChart3 className="h-8 w-8 text-[#d15642]" />,
      title: "Volume Estimation",
      description: "Get accurate volume measurements to assist with clinical assessment",
    },
    {
      icon: <Zap className="h-8 w-8 text-[#d15642]" />,
      title: "Fast Processing",
      description: "Results delivered in seconds to streamline your workflow",
    },
  ]

  return (
    <div className="space-y-8">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold mb-4">How It Works</h2>
        <p className="text-slate-300 max-w-2xl mx-auto">
          AlphaPolyp uses state-of-the-art AI to detect polyps in endoscopy images and estimate their volume, providing
          valuable data for clinical assessment.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {features.map((feature, index) => (
          <Card key={index} className="p-6 bg-slate-800 border-slate-700 hover:bg-slate-750 transition-colors">
            <div className="mb-4">{feature.icon}</div>
            <h3 className="text-xl font-medium mb-2">{feature.title}</h3>
            <p className="text-slate-300">{feature.description}</p>
          </Card>
        ))}
      </div>
    </div>
  )
}
