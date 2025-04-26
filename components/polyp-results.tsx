"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Clock, Box, Ruler } from "lucide-react"
import Image from "next/image"

interface PolypResultsProps {
  originalImage: string
  results: {
    volume: number
    dimensions: number[]
    segmentationPath?: string
    processingTime: number
    boundingBoxes: {
      id: number
      confidence: number
    }[]
  }
  fileName: string
}

export function PolypResults({ originalImage, results, fileName }: PolypResultsProps) {
  const subjectName = fileName.split(".")[0]

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
          Volume: {results.volume.toFixed(2)}cm³
        </Badge>

        <Badge variant="outline" className="flex items-center gap-1 text-sm py-1.5">
          <Clock className="h-4 w-4" />
          Processed in {results.processingTime.toFixed(1)}s
        </Badge>
      </div>

      {/* Display the result image with segmentation */}
      <div className="relative rounded-lg overflow-hidden border border-slate-700 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {/* Original Image */}
          <div className="relative aspect-video">
            <Image
              src={originalImage || "/placeholder.svg"}
              alt="Original polyp image"
              fill
              className="object-contain"
            />
          </div>

          {/* Result Image with Segmentation */}
          <div className="relative aspect-video bg-slate-900">
            {results.segmentationPath ? (
              <Image
                src={results.segmentationPath || "/placeholder.svg"}
                alt="Polyp segmentation result"
                fill
                className="object-contain"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                Segmentation not available
              </div>
            )}

            {/* Overlay text */}
            <div className="absolute top-2 left-2 right-2 text-white text-sm">
              <div>Subject: {subjectName}</div>
              <div>
                Volume: {results.volume.toFixed(2)} | Dims: {results.dimensions.map((d) => d.toFixed(2)).join(", ")}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        <h4 className="font-medium">Detailed Results:</h4>
        <div className="bg-slate-900 rounded-lg p-4 overflow-auto max-h-60">
          <pre className="text-xs text-slate-300">
            {JSON.stringify(
              {
                volume: results.volume,
                dimensions: results.dimensions,
                confidence: results.boundingBoxes[0]?.confidence || 0,
                processingTime: results.processingTime,
              },
              null,
              2,
            )}
          </pre>
        </div>
      </div>
    </Card>
  )
}
