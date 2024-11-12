## Multi modal RAG index creation from images in PDF

This sample application is based off the GitHub Sample located [here](!https://github.com/Azure-Samples/document-intelligence-code-samples/blob/main/Python(v4.0)/Retrieval_Augmented_Generation_(RAG)_samples/sample_figure_understanding.ipynb)

A few minor changes were made to the above in the sample

This application shows how to parse large PDF Documents that contain graphs, charts interspersed with text information.
Azure Document intelligence client is used to extract the details from each page in the PDF. When a chart is encountered as a part of the figures section in the client response document, gpt-4o model is used to describe the graphs in detail, extract trends, analysis, along with an extract of the data in tabular form.

The output of this program is a Markdown document that can then be vectorised and consumed in a Gen AI powered application