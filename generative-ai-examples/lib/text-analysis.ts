// Helper function to analyze text
export function analyzeText(text: string) {
  // Clean text
  const cleanText = text.toLowerCase().replace(/[^\w\s]/g, "")
  const words = cleanText.split(/\s+/).filter((word) => word.length > 0)

  // Count sentences (simple approximation)
  const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0)

  // Calculate metrics
  const wordCount = words.length
  const avgWordLength = words.length > 0 ? words.reduce((sum, word) => sum + word.length, 0) / words.length : 0
  const uniqueWords = new Set(words).size
  const lexicalDiversity = wordCount > 0 ? uniqueWords / wordCount : 0
  const avgSentenceLength = sentences.length > 0 ? wordCount / sentences.length : 0

  return {
    wordCount,
    avgWordLength,
    uniqueWords,
    lexicalDiversity,
    avgSentenceLength,
    sentenceCount: sentences.length,
  }
}
