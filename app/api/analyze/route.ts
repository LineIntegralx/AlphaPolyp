import { NextResponse } from "next/server"
import { saveImage, predictPolyp } from "@/lib/model-utils"

export async function POST(req: Request) {
  try {
    const formData = await req.formData()

    // Save the uploaded image
    const imagePath = await saveImage(formData)

    // Process the image with the model
    const results = await predictPolyp(imagePath)

    return NextResponse.json({
      success: true,
      imagePath,
      results,
    })
  } catch (error) {
    console.error("Error analyzing image:", error)
    return NextResponse.json({ success: false, error: "Failed to analyze image" }, { status: 500 })
  }
}
