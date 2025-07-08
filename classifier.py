import pandas as pd
from transformers import pipeline

# Step 1: Load the CSV file
df = pd.read_csv("student_feedback.csv")

# Step 2: Load FLAN-T5 model
classifier = pipeline("text2text-generation", model="google/flan-t5-small")

# Step 3: Define how we ask the model to classify feedback
def make_prompt(feedback):
    prompt = (
        "Classify the following student feedback into one of these categories: "
        "Academics, Facilities, Administration, Staff, Others.\n\n"
        "Feedback: \"The teachers were really helpful.\"\n"
        "Category: Staff\n\n"
        "Feedback: \"The fans in the classroom were not working.\"\n"
        "Category: Facilities\n\n"
        "Feedback: \"The syllabus was too hard.\"\n"
        "Category: Academics\n\n"
        f"Feedback: \"{feedback}\"\n"
        "Category:"
    )
    return prompt

# Step 4: Run model on each feedback
categories = []
for feedback in df["Feedback"]:
    prompt = make_prompt(feedback)
    result = classifier(prompt, max_new_tokens=10)[0]['generated_text']
    categories.append(result.strip())  # Clean extra spaces

# Step 5: Add the category to the original table
df["Category"] = categories

# Step 6: Save the output to a new CSV file
df.to_csv("classified_feedback.csv", index=False)

print("✅ Done! Check 'classified_feedback.csv' for results.")
