import { AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';

async function main() {
  console.log("Loading model...");
  const model_id = 'Xenova/ms-marco-MiniLM-L-6-v2';
  const tokenizer = await AutoTokenizer.from_pretrained(model_id);
  const model = await AutoModelForSequenceClassification.from_pretrained(model_id);
  
  const query = "What is the capital of France?";
  const documents = [
      "Paris is the capital of France.",
      "The sky is blue."
  ];

  console.log("Tokenizing...");
  // Try passing text_pair as options object or directly
  const inputs1 = await tokenizer(query, { text_pair: documents[0], padding: true, truncation: true });
  const inputs2 = await tokenizer(query, { text_pair: documents[1], padding: true, truncation: true });
  
  console.log("Ranking...");
  const out1 = await model(inputs1);
  const out2 = await model(inputs2);
  
  console.log("Score 1:", out1.logits.data);
  console.log("Score 2:", out2.logits.data);
}

main().catch(console.error);
