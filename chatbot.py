import os
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("ChatGPT is ready to converse! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Exiting conversation. Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name
    )
    
    reply = response.choices[0].message.content.strip()
    print("ChatGPT:", reply)
    
    messages.append({"role": "assistant", "content": reply})





    
# Define test cases
test_cases = [
    { #0
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 50,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    { #1
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 20,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 10,
        "No_of_Records": 500,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "trends",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 2,
        "No_of_Records": 150,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "ND",
        "No_of_Attributes": 4,
        "No_of_Records": 600,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "comparison",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 5,
        "No_of_Records": 750,
        "Primary_Variable (Data Type)": "geographical",
        "Task (Purpose)": "relationship",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 7,
        "No_of_Records": 350,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "comparison",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "ND",
        "No_of_Attributes": 9,
        "No_of_Records": 1100,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "relationship",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 3,
        "No_of_Records": 300,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "relationship",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 6,
        "No_of_Records": 850,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "trends",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 8,
        "No_of_Records": 1300,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "trends",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 10,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },  # Expected: Pie Chart
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 10,
        "No_of_Records": 5000,
        "Primary_Variable (Data Type)": "geographical",
        "Task (Purpose)": "comparison",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "Hierarchical",
        "No_of_Attributes": 20,
        "No_of_Records": 2500,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "comparison",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 9,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 2,
        "No_of_Records": 500,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "relationship",
        "Target Audience": "Expert"
    },  # Expected: Map
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 3,
        "No_of_Records": 300,
        "Primary_Variable (Data Type)": "continuous",
        "Task (Purpose)": "trends",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "2D",
        "No_of_Attributes": 3,
        "No_of_Records": 500,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "relationship",
        "Target Audience": "Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 9,
        "Primary_Variable (Data Type)": "categorical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Non-Expert"
    },

    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 35,
        "Primary_Variable (Data Type)": "ordinal",
        "Task (Purpose)": "tredistributionnds",
        "Target Audience": "Non-Expert"
    },
    {
        "Data_Dimensions": "1D",
        "No_of_Attributes": 1,
        "No_of_Records": 60,
        "Primary_Variable (Data Type)": "geographical",
        "Task (Purpose)": "distribution",
        "Target Audience": "Expert"
    }
]



# Test with predefined test cases
predictions = []
for test_case in test_cases:
    prediction, top_3 = get_prediction(test_case)
    predictions.append((test_case, prediction, top_3))

# Display results
df_results = pd.DataFrame(predictions, columns=["Test Case", "Predicted Chart", "Top 3 Predictions"])
print(df_results)



