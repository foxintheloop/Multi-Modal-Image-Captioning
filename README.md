# Multi-Modal Image Captioning

This project evaluates different approaches to building an AI system that can automatically generate natural-language descriptions of images (image captioning). It's particularly aimed at researchers and developers working on accessibility technology, as image captioning can help make visual content accessible to visually impaired users.

This study specifically compares:
- Different AI model configurations and optimization strategies
- Various techniques for converting images and text into formats the AI can understand
- Different training approaches on both local computers and cloud systems

This study tested 180 different configurations using the Flickr30k dataset (which contains 31,000 images, each with 5 human-written descriptions) and compared their custom-built system against BLIP, a leading pre-trained model. Studies Findings:

- The CLIP vectorization method worked best for processing images
- A specific combination of optimization settings (Adagrad optimizer with StepLR scheduler) consistently performed well
- Cloud computing was necessary for effective training, as local machines had significant limitations

The project's findings are particularly valuable for:

- AI researchers working on multi-modal systems (systems that combine image and text processing)
- Developers implementing image captioning in real-world applications
- Organizations looking to make their visual content more accessible
- Teams working with limited computational resources who need to make informed decisions about model configuration
