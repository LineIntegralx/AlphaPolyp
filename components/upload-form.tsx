"use client"

import type React from "react"

import { useState, useRef } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { UploadCloud, X, Loader2 } from "lucide-react"
import { PolypResults } from "@/components/polyp-results"

export function UploadForm() {
  const [image, setImage] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<any | null>(null)
  const [uploadedImagePath, setUploadedImagePath] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = (event) => {
        setImage(event.target?.result as string)
        setResults(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = (event) => {
        setImage(event.target?.result as string)
        setResults(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const clearImage = () => {
    setImage(null)
    setFileName("")
    setResults(null)
    setUploadedImagePath(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const analyzeImage = async () => {
    if (!image) return

    setIsAnalyzing(true)

    try {
      // Create a FormData object to send the image
      const formData = new FormData()

      // Convert base64 to blob
      const response = await fetch(image)
      const blob = await response.blob()

      // Add the image to the form data
      formData.append("image", blob, fileName || "image.jpg")

      // Send the image to the API
      const apiResponse = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      })

      const data = await apiResponse.json()

      if (data.success) {
        setUploadedImagePath(data.imagePath)
        setResults(data.results)
      } else {
        console.error("Error analyzing image:", data.error)
        alert("Failed to analyze image. Please try again.")
      }
    } catch (error) {
      console.error("Error analyzing image:", error)
      alert("Failed to analyze image. Please try again.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="w-full max-w-3xl">
      <Card className="p-6 bg-slate-800 border-slate-700">
        {!image ? (
          <div
            className="border-2 border-dashed border-slate-600 rounded-lg p-12 flex flex-col items-center justify-center cursor-pointer hover:border-slate-500 transition-colors"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <UploadCloud className="h-12 w-12 text-white mb-4" />
            <h3 className="text-xl font-medium mb-2">Upload Polyp Image</h3>
            <p className="text-slate-400 text-center mb-4">Drag and drop your image here or click to browse</p>
            <Button variant="outline">Select Image</Button>
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-xl font-medium">{fileName}</h3>
              <Button variant="ghost" size="icon" onClick={clearImage}>
                <X className="h-5 w-5" />
              </Button>
            </div>

            <div className="relative aspect-square max-h-[500px] overflow-hidden rounded-lg">
              <Image src={image || "/placeholder.svg"} alt="Uploaded polyp image" fill className="object-contain" />
            </div>

            <Button
              className="w-full bg-[#d15642] hover:bg-[#b34535] text-white"
              onClick={analyzeImage}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Analyze Image"
              )}
            </Button>
          </div>
        )}
      </Card>

      {results && <PolypResults originalImage={uploadedImagePath || image} results={results} fileName={fileName} />}
    </div>
  )
}
