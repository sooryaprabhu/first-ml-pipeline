import openai
import os


def explain_prediction(features, predicted_price):
    # Get API key from environment variable
    # NEVER hardcode this — always from environment
    client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # System prompt — instructions to the AI
    # This tells the AI WHO it is and HOW to behave
    system_prompt = """You are a helpful real estate expert AI assistant.
You explain house price predictions in simple, clear English.
You are speaking to someone who may not understand technical terms.
Keep your explanation to 3-4 sentences maximum.
Always be friendly and professional."""

    # User prompt — the actual question with data
    # We inject the real house data and prediction here
    user_prompt = f"""A machine learning model predicted this house price:

House Details:
- Median Income in area: ${features['MedInc'] * 10000:,.0f} per year
- House Age: {features['HouseAge']} years
- Average Rooms: {features['AveRooms']:.1f}
- Average Bedrooms: {features['AveBedrms']:.1f}
- Population in block: {features['Population']:,.0f}
- Average Occupants: {features['AveOccup']:.1f}
- Location: {features['Latitude']:.2f}N, {features['Longitude']:.2f}W

Predicted Price: ${predicted_price:,.0f}

Please explain in simple English why this house might be worth 
this price based on these features. Focus on the most important factors."""

    # Make the API call to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",

        # Temperature controls creativity vs consistency
        # 0.0 = very consistent, always same answer
        # 1.0 = very creative, different every time
        # 0.7 = good balance for explanations
        temperature=0.7,

        # max_tokens = maximum length of response
        # 1 token ≈ 4 characters
        # 200 tokens ≈ 150 words
        max_tokens=200,

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Extract the text from the response
    explanation = response.choices[0].message.content
    return explanation


if __name__ == "__main__":
    # Test the explainer with sample house data
    test_features = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

    test_price = 415194.31

    print("Generating AI explanation...")
    explanation = explain_prediction(test_features, test_price)
    print("\nAI Explanation:")
    print(explanation)
