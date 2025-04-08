import define1 from "./26670360aa6f343b@226.js";
import define2 from "./a2166040e5fb39a6@229.js";

function _1(md){return(
md`# Results for MRI Scan Image Classification for Alzheimer's Onset `
)}

function _2(md){return(
md`### Import data + packages`
)}

function _cnn1(FileAttachment){return(
FileAttachment("Aggregated_Results_CNN_v1.csv").csv()
)}

function _cnn2(FileAttachment){return(
FileAttachment("Aggregated_Results_CNN_v2.csv").csv()
)}

function _vit1(FileAttachment){return(
FileAttachment("Aggregated_Results_transformer_v1.csv").csv()
)}

function _vit2(FileAttachment){return(
FileAttachment("Aggregated_Results_transformer_v2.csv").csv()
)}

function _best(FileAttachment){return(
FileAttachment("Best_revised.csv").csv()
)}

function _10(printTable,best){return(
printTable(best.slice(0,5))
)}

function _11(md){return(
md`## Best Models Summary file visualizations`
)}

function _12(vl,best){return(
vl.markBar()
  .data(best)
  .encode(
    vl.x().fieldN('Model').title('Model').axis({ labelAngle: -30 }),
    vl.y().fieldQ('Accuracy').title('Accuracy Score'),
    vl.color().fieldN('Preprocessing').title('Preprocessing')
              .legend({ orient: 'right', titleFontSize: 12 }), // preprocessing color coding
    vl.column().fieldN('Balanced').title('') 
      .header({
        labelExpr: "datum.value == 'Yes' ? 'Balanced data' : 'Unbalanced data'" // Relabel values for subplots
      }),
    vl.tooltip().fieldN('Model_type')
  )
  .width(200) // sub-chart width
  .height(400)
  .title({
    text: 'Accuracy of Best Performing Models', 
    anchor: 'middle', // Center title
    fontSize: 14,
    offset: 20 // vertical spacing
  })
   

  .render()
)}

function _13(vl,best){return(
vl.markBar()
  .data(best)
  .encode(
    vl.x().fieldN('Model').title('Model').axis({ labelAngle: -30 }),
    vl.y().fieldQ('AUC').title('AUC Score'),
    vl.color().fieldN('Preprocessing').title('Preprocessing')
              .legend({ orient: 'right', titleFontSize: 12 }),
    vl.column().fieldN('Balanced').title('') 
      .header({
        labelExpr: "datum.value == 'Yes' ? 'Balanced data' : 'Unbalanced data'" // Relabel values for subplots
      }),
    vl.tooltip().fieldN('Model_type')
  )
  .width(200) // sub-chart width
  .height(400)
  .title({
    text: 'AUC of Best Performing Models', 
    anchor: 'middle', // Center title
    fontSize: 14,
    offset: 20 // vertical spacing
  })
   

  .render()
)}

function _14(best){return(
best.map(d => d.Balanced)
)}

function _15(vl,best){return(
vl.markPoint({ size: 100 })
  .data(best)
  .encode(
    vl.x().fieldQ('Accuracy').title('Accuracy').scale({ domain: [0.4, 1] }), // set bounds 
    vl.y().fieldQ('AUC').title('AUC').scale({ domain: [0.7, 1] }),
    vl.color().fieldN('Model').title('Model').scale({ scheme: "category10" }),
    vl.shape().fieldN('Balanced'),
    vl.tooltip([vl.tooltip().fieldN('Model_type'), vl.tooltip().fieldQ('F1-Score')])
  )
  .width(450)
  .height(300)
  .title('Accuracy vs. AUC of Best Performing Models')
   

  .render()
)}

function _16(vl,best){return(
vl.markBar()
  .data(best)
  .encode(
    vl.y().fieldQ('Accuracy').title('Accuracy Score'),
    vl.x().fieldN('Model').title('Model').axis({ labelAngle: -10 }),
    vl.color().fieldN('Model_type'), // Color by Model_type
    vl.xOffset().fieldN('Model_type'),
    vl.tooltip().fieldN('Model_type')
  )
  .width(800)
  .height(300)
  .title('Accuracy of Best Performing Models')
.render()
)}

function _17(md){return(
md`## EDA Visualizations `
)}

function _traintestsplit(){return(
[
  { "category": "Train Images", "value": 5120 },
  { "category": "Test Images",  "value": 1280 }
]
)}

function _19(vl,traintestsplit){return(
vl.markArc({ innerRadius: 0, outerRadius: 130 })
  .data(traintestsplit)
  .transform(
    vl.joinaggregate([{ op: 'sum', field: 'value', as: 'TotalCount' }]),
    vl.calculate('datum.value / datum.TotalCount').as('Percentage') // find percentage of total
  )
  .encode(
    vl.theta().fieldQ('value').title('count'), // set number of rows as value
    vl.color().fieldN('category').title('') // assign to data split 
       .scale({scheme: "paired" }),
    vl.tooltip([ // display percentages upon hover
      vl.tooltip().fieldQ('value').title('Count'),
      vl.tooltip().fieldQ('Percentage').title('Percentage').format('.1%')
    ])
  )
  // plot settings
  .width(300)
  .height(300)
  .title('Train Test Data Split')

  .render()
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    ["Aggregated_Results_transformer_v1.csv", {url: new URL("./files/7f6677e06386a80996b70df5fb79a98910f4bda418a7912d692cbe2400fe0f68bc17ac7ae1aef2dbf21a7da140d1e23de872d10253df86f249ddd828cead9340.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["Aggregated_Results_CNN_v2.csv", {url: new URL("./files/35d0421ccba9037cd5552e95d8c6782ea8d4d8670d026932bf01b73bd218bcee60b5b1a2a8d46796a1cfe4057c2cc99e6bc2687633feecf1fc2991fce1a959d1.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["Aggregated_Results_transformer_v2.csv", {url: new URL("./files/c683b502d1072d2ce56cc61e559a0da6c5d4e7b36386c2658f3ab8f30b7a208b43de9fe7a9758f34abe758b2494d57a42ee812e41defa05a727170df15915771.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["Aggregated_Results_CNN_v1.csv", {url: new URL("./files/b283c20d06c8b1d5902cb2fa84fdea009215d5614d832ce936637ef91c2e3b3813512f6f7f5d465d9e00ae1166b78d526e9c112815b748f74fb12f3b54efcfc8.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["Best_revised.csv", {url: new URL("./files/53d04041f4224bd676d8abf53d0ba0d3bc3da816e3abc337b7afb0571ab8c02a2419ea113a49dfdd0f5d7ae0a15ab7c7374348659241b5933c171e5ee8320df7.csv", import.meta.url), mimeType: "text/csv", toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  const child1 = runtime.module(define1);
  main.import("vl", child1);
  const child2 = runtime.module(define2);
  main.import("printTable", child2);
  main.variable(observer("cnn1")).define("cnn1", ["FileAttachment"], _cnn1);
  main.variable(observer("cnn2")).define("cnn2", ["FileAttachment"], _cnn2);
  main.variable(observer("vit1")).define("vit1", ["FileAttachment"], _vit1);
  main.variable(observer("vit2")).define("vit2", ["FileAttachment"], _vit2);
  main.variable(observer("best")).define("best", ["FileAttachment"], _best);
  main.variable(observer()).define(["printTable","best"], _10);
  main.variable(observer()).define(["md"], _11);
  main.variable(observer()).define(["vl","best"], _12);
  main.variable(observer()).define(["vl","best"], _13);
  main.variable(observer()).define(["best"], _14);
  main.variable(observer()).define(["vl","best"], _15);
  main.variable(observer()).define(["vl","best"], _16);
  main.variable(observer()).define(["md"], _17);
  main.variable(observer("traintestsplit")).define("traintestsplit", _traintestsplit);
  main.variable(observer()).define(["vl","traintestsplit"], _19);
  return main;
}
