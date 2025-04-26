"use client"

import { useRef, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Clock, Box, Ruler } from "lucide-react"

interface BoundingBox {
  id: number
  x: number
  y: number
  width: number
  height: number
  confidence: number
  volume: number
}

interface ImageResultProps {
  image: string | null
  results: {
    boundingBoxes: BoundingBox[]
    totalVolume: number
    processingTime: number
  }
}

export function ImageResult({ image, results }: ImageResultProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current || !image) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = "anonymous"
    img.src = image

    img.onload = () => {
      // Set canvas dimensions to match the image
      canvas.width = img.width
      canvas.height = img.height

      // Draw the image
      ctx.drawImage(img, 0, 0)

      // Draw bounding boxes
      results.boundingBoxes.forEach((box) => {
        const x = box.x * canvas.width
        const y = box.y * canvas.height
        const width = box.width * canvas.width
        const height = box.height * canvas.height

        // Draw rectangle
        ctx.strokeStyle = "#d15642"
        ctx.lineWidth = 3
        ctx.strokeRect(x, y, width, height)

        // Draw label background
        ctx.fillStyle = "rgba(209, 86, 66, 0.8)"
        ctx.fillRect(x, y - 25, 140, 25)

        // Draw label text
        ctx.fillStyle = "white"
        ctx.font = "14px Arial"
        ctx.fillText(`Polyp: ${(box.confidence * 100).toFixed(0)}% - ${box.volume.toFixed(1)}cm³`, x + 5, y - 8)
      })
    }
  }, [image, results])

  return (
    <Card className="p-6 mt-6 bg-slate-800 border-slate-700">
      <h3 className="text-xl font-medium mb-4">Analysis Results</h3>

      <div className="flex flex-wrap gap-4 mb-4">
        <Badge variant="outline" className="flex items-center gap-1 text-sm py-1.5">
          <Box className="h-4 w-4" />
          {results.boundingBoxes.length} polyp{results.boundingBoxes.length !== 1 ? "s" : ""} detected
        </Badge>

        <Badge variant="outline" className="flex items-center gap-1 text-sm py-1.5">
          <Ruler className="h-4 w-4" />
          Total volume: {results.totalVolume.toFixed(1)}cm³
        </Badge>

        <Badge variant="outline" className="flex items-center gap-1 text-sm py-1.5">
          <Clock className="h-4 w-4" />
          Processed in {results.processingTime.toFixed(1)}s
        </Badge>
      </div>

      <div className="relative rounded-lg overflow-hidden border border-slate-700">
        <canvas ref={canvasRef} className="w-full h-auto" />
      </div>

      <div className="mt-4 space-y-2">
        <h4 className="font-medium">Detailed Results:</h4>
        <div className="bg-slate-900 rounded-lg p-4 overflow-auto max-h-60">
          <pre className="text-xs text-slate-300">{JSON.stringify(results, null, 2)}</pre>
        </div>
      </div>
    </Card>
  )
}
