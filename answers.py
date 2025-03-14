questions = [
    "How do you combine multiple adapters?",
    "What are the benefits of using PEFT?",
    "How can I fine-tune a model with PEFT?",
    "What are the different types of adapters available in PEFT?",
    "Is PEFT compatible with other libraries like Transformers?"
]

# Lista de respuestas
answers = [
    "PEFT allows you to combine multiple adapters by using the `Sequential` or `Parallel` modules.",
    "Some benefits of using PEFT include reduced memory usage and faster training times.",
    "You can fine-tune a model with PEFT by using the `get_peft_model` function.",
    "PEFT offers various adapter types like LoRA, Prefix Tuning, and Compacter.",
    "Yes, PEFT is compatible with other libraries like Transformers and Accelerate."
]

for question in questions:
    # Seleccionar una respuesta
    answer = choice(answers)

    # Salida del modelo
    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
