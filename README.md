Implementación de varios descriptores **globales** que no involucran machine learning en C++. Rendimiento de F1-Score de $\approx 0.77$ en datasets de más de $10.000$ imágenes. 

Para compilar basta ejecutar `make all`. El testeo fue implementado en MacOS 14.0 con clang. 

Los mejores rendimientos se obtienen en: 
1. Imágenes de calidad reducida
2. Imágenes recortadas
3. Imágenes fragmentadas
4. Imágenes con cambios de perspectiva
5. Imágenes con filtros Gamma


