from app.ml.sentiment_analyzer import sentiment_analyzer

print("Testing sentiment analyzer...")
print("\nFetching and analyzing Bitcoin news...\n")

result = sentiment_analyzer.get_overall_sentiment("BTCUSDT")

print(f"Signal: {result['signal']}")
print(f"Sentiment Score: {result['sentiment_score']:.2f}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Articles Analyzed: {result['articles_analyzed']}")
print("\nTop Headlines:")
for i, headline in enumerate(result['headlines'], 1):
    print(f"{i}. [{headline['sentiment'].upper()}] {headline['headline']}")