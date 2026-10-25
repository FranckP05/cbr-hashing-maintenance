# Stabilisation de la démo CBR

## But

Rendre la démo fiable pour une soutenance : les visualisations doivent apparaître
dès la première analyse, conserver leurs dimensions après chaque interaction et
rester lisibles sans dépendre d'un accès réseau au moment de la présentation.

## Diagnostic confirmé

La démo renvoie un fragment HTML complet dans `#demo-dynamic` à chaque action
HTMX. Les scripts Chart.js sont inclus dans ce fragment. Ils enregistrent leurs
gestionnaires `htmx:afterSwap` après que l'événement du swap courant a déjà eu
lieu. La première visualisation peut donc ne pas être créée ; un swap ultérieur,
tel que l'affichage de la maintenance, la crée à contretemps. Le même mécanisme
reconstruit inutilement les deux graphiques.

Les dépendances frontales sont en outre chargées depuis des CDN sans copie locale
ni interface de repli. Enfin, la visualisation de maintenance ne contient que
deux proportions : une bibliothèque de graphiques côté navigateur n'est pas
nécessaire pour cette information.

## Architecture retenue

```
Flask + code expérimental exp2
          |
     Jinja + HTMX
          |
  static/js/charts.js  ----> Chart.js (benchmark seulement)
          |
  HTML/CSS/SVG --------> comparaison maintenance et icônes
```

- Flask, Jinja et HTMX sont conservés : le parcours séquentiel de la démo ne
  justifie pas une SPA ni une migration de framework.
- Le cycle de vie Chart.js est regroupé dans un unique script statique chargé
  une fois. Il détruit les instances avant le remplacement de leur conteneur et
  initialise les nouveaux canevas depuis les hooks HTMX globaux.
- Les données du benchmark sont transmises sous forme JSON dans le HTML ; le
  script statique ne contient aucune valeur Jinja et aucun script n'est injecté
  dans les fragments.
- Le graphique d'impact CBM devient une comparaison HTML/SVG accessible de deux
  barres. Il ne peut donc ni déclencher de boucle de redimensionnement ni
  perturber le benchmark.
- Les bibliothèques nécessaires à l'interaction sont servies localement et
  leurs versions sont figées. Le chargement de la page ne dépend plus d'un CDN.
- Les emojis et symboles décoratifs sont remplacés par du texte ou de petites
  icônes SVG de style Phosphor (trait fin, `viewBox` 24 × 24). Le texte porte
  toujours le sens pour l'accessibilité.

## Changements prévus

1. Déplacer les scripts CDN utilisés par la démo dans `demo/static/vendor/` et
   les charger depuis `base.html`.
2. Créer `demo/static/js/charts.js`, avec un registre d'instances Chart.js et
   les hooks HTMX globaux `beforeSwap` et `afterSettle`.
3. Remplacer les deux blocs JavaScript de `flow.html` par un canevas benchmark
   doté de données JSON, puis par une comparaison de maintenance rendue en HTML.
4. Ajouter les styles dédiés aux icônes, aux graphiques et à leur message de
   repli ; ne plus utiliser de styles en ligne pour ces éléments.
5. Corriger la route maintenance obsolète ou l'aligner sur le fragment qui
   existe réellement.
6. Ajouter des tests de rendu Flask et un scénario navigateur lorsque
   l'environnement de test le permet.

## Critères d'acceptation

- À froid, la première analyse affiche le benchmark sans deuxième clic.
- Dix activations/réinitialisations de la maintenance ne changent ni les
  dimensions ni le nombre d'instances du benchmark.
- L'impact de maintenance s'affiche sans Chart.js.
- La page reste fonctionnelle pour le flux principal lorsque les CDN externes
  sont inaccessibles.
- Aucun emoji décoratif ne demeure dans l'interface de la démo.
- Les routes Flask concernées répondent sans erreur de template.

## Alternatives écartées

Une migration vers Django Unicorn ne traite pas ce défaut de cycle de vie :
Unicorn est lié à Django et utilise lui aussi un runtime JavaScript pour les
requêtes et la mise à jour du DOM. Une SPA React/Vue/Svelte augmenterait
fortement le périmètre sans gain approprié pour une démo de soutenance. Une
visualisation serveur totalement statique reste une option de repli, mais le
benchmark Chart.js conservé ci-dessus offre les infobulles utiles avec un seul
module JavaScript contrôlé.
