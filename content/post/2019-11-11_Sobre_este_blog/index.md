---
title: "Sobre este blog"
subtitle: "Un poco sobre mí, el origen y la motivación detrás de este blog. 🚀"
date: 2019-11-11
lastmod: 2020-04-11
summary: Un poco sobre mí, el origen y la motivación detrás de este blog.
authors:
  - admin
categories: [Blog]
draft: false
# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Placement options: 1 = Full column width, 2 = Out-set, 3 = Screen-width
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
image:
  placement: 1
  caption: "Image credit: [**Unsplash**](https://unsplash.com/photos/_uM5_nG2ssc)"
  focal_point: ""
  preview_only: false
---

Hoy comienzo este proyecto personal: **mi primer blog**.

Soy bastante nuevo en el campo de Data Science, sin embargo el mundo de los datos no me es para nada ajeno, ya que desde hace casi 5 años me desempeño realizando tareas de [Business Intelligence](https://es.wikipedia.org/wiki/Inteligencia_empresarial) y [Reporting](https://en.wikipedia.org/wiki/Data_reporting).

Mi objetivo a mediano plazo es continuar formándome como Data Scientist y desarrollarme en áreas donde se desarrollen modelos predictivos y algoritmos de aprendizaje automático.

Durante esta etapa de formación considero algo fundamental desarrollar pequeños proyectos para aplicar y consolidar los nuevos conocimientos, a la vez que me permite mostrar mi trabajo y habilidades.

Como parte de estos proyectos surgió esta idea de crear un blog / portfolio, ya que escribir me obliga a ordenar mis pensamientos y profundizarlos hasta llegar al punto en que lo pueda explicar de manera concisa y sencilla. Como punto adicional, también me sirve de archivo para encontrar cosas que a veces se olvidan... 😋

{{< figure src="blogging.jpg" >}}

### Creando el blog

Una vez decidido el _qué_, empezó la etapa del _cómo_.

Sabía que quería escribir un blog, pero también que es una actividad que demanda una buena cantidad de tiempo. Quería utilizar una plataforma que me hiciera fácil las cosas y me permitiera hacer foco solo en el contenido.

En el pasado había trabajado un poco con [WordPress](https://es.wordpress.com/) y no me habia resultado del todo cómodo. No me daba la libertad de publicar de la manera que yo quería.
Pensé también en incursionar con _Static Site Generators_ como [Gatsby](https://www.gatsbyjs.org/) o [Next.JS](https://nextjs.org/) o alguna otra plataforma ese estilo, pero requiere un tiempo considerable para aprender a utilizarlo, hacer el desarrollo y luego mantenerlo. No, ya intenté eso y no funcionó...

Luego pensé en [Medium](https://medium.com), una plataforma que me resulta muy agradable y placentera estéticamente, pero el modelo de negocio que adoptaron los últimos años (Paywall) me fueron alejando. Además, pensando a largo plazo, quería tener mayor control sobre el contenido.

Entonces recordé que hace unos años había experimentado un poco con **Jekyll** y **GitHub Pages**.

[Jekyll](https://jekyllrb.com/) es generador de sitios web estáticos con capacidades de blogging, desarrollado en Ruby. Su principal característica es que, en lugar de utilizar bases de datos, Jekyll toma contenido en formato [Markdown](https://es.wikipedia.org/wiki/Markdown) y produce como resultado sitios web estáticos listos para ser subidos a servidores de contenido estático como Apache, Nginx, etc.

Como plus, Jekyll es el motor de [GitHub Pages](https://pages.github.com/) una funcionalidad de GitHub que permite a los usuarios hospedar sitios web desde sus repositorios.

Si a todo esto, le sumamos que encontré un _theme_ muy flexible y estéticamente bastante similar a Medium, ~~bingo! Ya tengo todo lo necesario...~~ 🤔 (Ver Update)

El theme en cuestión es [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/), un tema flexible y minimalista ideal para el desarrollo de blogs y portfolios. ¡Justo lo que andaba necesitando!

{{% alert light %}}

##### Update Abril 2020:

Luego de utilizar Jekyll unos meses comencé a notar que cuantos más posts escribía, más lento se hacía todo...
Googleando y leyendo un poco, me di cuenta que no era el único al que le pasaba esto. Empecé a buscar alternativas. Ya me había acostumbrado a la plataforma, quería seguir escribiendo los posts en Markdown, que se realizara el deploy con un simple push al repositorio y en la medida de lo posible sin tener costos de hosting. Esas eran las condiciones para cambiar, y encontré algo mejor: [HUGO](https://gohugo.io/) + [Netlify](https://www.netlify.com/)

HUGO es un _static site generator_ que, al igual que Jekyll, utiliza Markdown para crear el contenido. La diferencia es que es muuuucho más rápido que Jekyll!!! 🚀.
Por otro lado, Netlify es una plataforma para hosting de aplicaciones web modernas, que permite conectar con un repositorio de Github y hacer un build+deploy del proyecto con cada push al repositorio. Y también es increíblemente rápido! Y lo mejor de todo, es que tienen un plan **gratuito** para proyectos pequeños, como este humilde blog!.

Todavía lo estoy probando, pero ya estoy enamorado de esta nueva plataforma. 💘 Veremos como sigue esta historia!

Las conclusiones a las que llegué aquí abajo 👇, son las mismas para esta nueva plataforma.
{{% /alert %}}

### Conclusión

Esta solución me permitió poner el funcionamiento el blog en muy pocas horas. Incluso personalizándolo estéticamente a mi gusto con unas configuraciones bastantes simples.

La creación de contenido la realizo directamente escribiendo en Markdown con [VS Code](https://code.visualstudio.com/) y un par de extensiones que son de mucha utilidad. Esto me permite utilizar las mismas herramientas que ya utilizaba y sin necesidad de explorar nada nuevo, concentrándome únicamente en el contenido.

Esto es todo para este primer post.

> La máquina ya está rodando, ahora solo resta mantenerla en movimiento!

Hasta la próxima!
