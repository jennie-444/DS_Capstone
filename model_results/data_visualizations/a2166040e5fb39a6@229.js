// https://observablehq.com/@uwdata/data-utilities@229
function _1(md){return(
md`# Data Utilities`
)}

function _2(md){return(
md`
Miscellaneous data utilities for Observable notebooks.
`
)}

function _3(md){return(
md`
## printTable(_data[, header, width]_)

Print an array of _data_ objects as an HTML-formatted table.

The optional _header_ array indicates which fields to include and in what order. The _header_ array values can be either a property name string or an object with the following properties:

- \`field\` - object property name
- \`title\` - [optional] column title to display
- \`align\` - [optional] alignment, one of \`'l'\`, \`'c'\`, or \`'r'\` (left, center, or right).

The optional _width_ parameter specifies the maximum width of the table, in pixels. If \`null\` or zero, the maximum width will be inherited from CSS. If \`undefined\` or not a number, Observable's \`width\` value will be used.
`
)}

function _printTable(width,printValue,html){return(
function printTable(data, header, maxWidth) {
  const Align = {
    l: 'left',
    c: 'center',
    r: 'right'
  };
  
  const cellStyle = 'padding: 1px 3px; white-space: nowrap;';
 
  header = header || (data[0] && Object.keys(data[0]));
  if (!header) return '';
  const fields = header.map(h => (h && h.field || h) + '');
  const styles = header.map(h => {
    return `style="text-align:${h && Align[h.align] || Align.l}; ${cellStyle}"`;
  });
  
  const lines = [];
  const maxw = maxWidth || maxWidth === undefined
    ? ` style="max-width:${+maxWidth || width}px"`
    : '';
  lines.push(`<table${maxw}>`);
  
  // header row
  lines.push('<thead><tr>');
  header.forEach((h, i) => {
    lines.push(`<th ${styles[i]}>${h && h.title || h}</th>`);
  });
  lines.push('</tr></thead>');
  
  // data rows

  lines.push('<tbody>');
  data.forEach(d => {
    lines.push('<tr>');
    fields.map((f, i) => {
      lines.push(`<td ${styles[i]}>${printValue(d[f])}</td>`);
    });
    lines.push('</tr>');
  });
  lines.push('</tbody>');
  
  lines.push('</table>');
  return html`<div style="overflow-x: auto;">${lines.join('')}</div>`;
}
)}

function _printValue(dateString){return(
function printValue(v) {
  if (v instanceof Date) {
    if (v.getHours() === 0 &&
        v.getMinutes() === 0 &&
        v.getSeconds() === 0 &&
        v.getMilliseconds() === 0)
    {
      return dateString(v.getFullYear(), v.getMonth()+1, v.getDate());
    } else if (v.getUTCHours() === 0 &&
               v.getUTCMinutes() === 0 &&
               v.getUTCSeconds() === 0 &&
               v.getUTCMilliseconds() === 0)
    {
      return dateString(v.getUTCFullYear(), v.getUTCMonth()+1, v.getUTCDate());
    } else {
      return v.toISOString();
    }
  } else {
    return JSON.stringify(v);
  }
}
)}

function _dateString(){return(
function dateString(year, month, date) {
  return `${year}-${(month < 10 ? '0' : '') + month}-${(date < 10 ? '0' : '') + date}`;
}
)}

function _7(md){return(
md`### Tests`
)}

function _testData(){return(
[
  {foo: 'a',  bar: 1,   baz: true,  date: new Date(2010, 0, 1)},
  {foo: 'b',  bar: 2,   baz: false, date: new Date(2010, 5, 15)},
  {foo: 'c',  bar: -1,  baz: null,  date: new Date(2010, 9, 31)},
  {foo: 'd',  bar: NaN, date: new Date(2010, 2, 10, 13, 45, 23)}
]
)}

function _9(printTable,testData){return(
printTable(testData)
)}

function _10(printTable,testData){return(
printTable(testData, [
  {field: 'foo', title: 'Foo', align: 'l'},
  {field: 'bar', title: 'Bar', align: 'c'},
  {field: 'baz', title: 'Baz', align: 'r'}
])
)}

function _11(printTable,testData){return(
printTable(testData.slice().reverse(), ['baz', 'bar'], '*')
)}

function _12(printTable){return(
printTable(['ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').reduce((o,c) => {
  o['column' + c] = 1;
  return o;
}, {})])
)}

function _13(printTable){return(
printTable([{
  text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mauris imperdiet diam quam, in molestie nibh tempor vitae. Proin finibus eget nunc et aliquam. Mauris vulputate et ante id egestas. Praesent finibus metus sed interdum accumsan. Praesent a luctus leo, at vulputate orci. Vivamus sit amet ligula efficitur, gravida nulla eget, pharetra sem. Suspendisse erat nunc, sollicitudin quis rhoncus in, maximus vitae elit. Proin sodales hendrerit cursus.",
  value: 5
}])
)}

function _14(md){return(
md`
## mdTable(_data[, header]_)

Print an array of _data_ objects as an HTML-formatted table that uses the width allotted for Markdown output.

The optional _header_ array indicates which fields to include and in what order. The _header_ array values can be either a property name string or an object with the following properties:

- \`field\` - object property name
- \`title\` - [optional] column title to display
- \`align\` - [optional] alignment, one of \`'l'\`, \`'c'\`, or \`'r'\` (left, center, or right).
`
)}

function _mdTable(printTable){return(
function mdTable(data, header) {
  return printTable(data, header, null);
}
)}

function _16(md){return(
md`
## fromColumns(_vectors_)
Generate an array of data objects given input _column_ vectors.
`
)}

function _fromColumns(d3){return(
function fromColumns(vectors) {
  const fields = Object.keys(vectors),
        nrows = d3.max(fields, _ => vectors[_].length),
        data = Array(nrows);
  
  for (let i=0; i<nrows; ++i) {
    data[i] = fields.reduce((o, _) => (o[_] = vectors[_][i], o), {});
  }
  
  return data;
}
)}

function _18(md){return(
md`### Tests`
)}

function _19(mdTable,fromColumns){return(
mdTable(fromColumns({
  u: 'abcdefgh',
  v: [2, 8, 3, 7, 5, 4, 6, 1]
}))
)}

function _20(md){return(
md`
## downloadLink(_linkText, file, content[, mimeType]_)
Generate an \`a\` link with a _linkText_ label that downloads _content_ to the _file_ name. The optional _mimeType_ specifies the [MIME type](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types/Complete_list_of_MIME_types), which defaults to \`"text/plain"\`.
`
)}

function _downloadLink(html){return(
function downloadLink(label, fileName, content, mimeType) {
  // first, create a content blob
  const blob = new Blob([content], {type: mimeType || 'text/plain'});
  const url = URL.createObjectURL(blob);
  
  // second, create a download link for that blob
  return html`<a href=${url} download="${fileName}">${label}</a>`;
}
)}

function _22(md){return(
md`### Tests`
)}

function _23(downloadLink){return(
downloadLink('Download Dummy Text!', 'dummy.txt', 'dummy')
)}

function _24(md){return(
md`
## unique(_data[, accessor]_)
Return a list of unique values in the _data_ array, optionally applying a provided _accessor_ function to extract values first.
`
)}

function _unique(){return(
function unique(data, accessor) {
  return Array.from(new Set(accessor ? data.map(accessor) : data));
}
)}

function _26(md){return(
md`### Tests`
)}

function _27(unique){return(
unique([1, 2, 1, 2, 3, 3])
)}

function _28(unique){return(
unique([{k:'a'}, {k:'b'}, {k:'c'}, {k:'c'}, {k:'a'}], d => d.k)
)}

function _29(md){return(
md`
## uniqueValid(_data[, accessor]_)
Return a sorted list of unique values in the _data_ array, optionally applying a provided _accessor_ function to extract values first. Removes any \`null\`, \`undefined\`, or \`NaN\` values from the unique value set.
`
)}

function _uniqueValid(unique){return(
function uniqueValid(data, accessor) {
  return unique(data, accessor)
     .filter(d => d != null && d === d)
     .sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
}
)}

function _31(md){return(
md`### Tests`
)}

function _32(uniqueValid){return(
uniqueValid([3, 1, 2, null, undefined, NaN, 1, 2, 3])
)}

function _33(md){return(
md`## Imports`
)}

function _d3(require){return(
require('d3-array')
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer()).define(["md"], _2);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer("printTable")).define("printTable", ["width","printValue","html"], _printTable);
  main.variable(observer("printValue")).define("printValue", ["dateString"], _printValue);
  main.variable(observer("dateString")).define("dateString", _dateString);
  main.variable(observer()).define(["md"], _7);
  main.variable(observer("testData")).define("testData", _testData);
  main.variable(observer()).define(["printTable","testData"], _9);
  main.variable(observer()).define(["printTable","testData"], _10);
  main.variable(observer()).define(["printTable","testData"], _11);
  main.variable(observer()).define(["printTable"], _12);
  main.variable(observer()).define(["printTable"], _13);
  main.variable(observer()).define(["md"], _14);
  main.variable(observer("mdTable")).define("mdTable", ["printTable"], _mdTable);
  main.variable(observer()).define(["md"], _16);
  main.variable(observer("fromColumns")).define("fromColumns", ["d3"], _fromColumns);
  main.variable(observer()).define(["md"], _18);
  main.variable(observer()).define(["mdTable","fromColumns"], _19);
  main.variable(observer()).define(["md"], _20);
  main.variable(observer("downloadLink")).define("downloadLink", ["html"], _downloadLink);
  main.variable(observer()).define(["md"], _22);
  main.variable(observer()).define(["downloadLink"], _23);
  main.variable(observer()).define(["md"], _24);
  main.variable(observer("unique")).define("unique", _unique);
  main.variable(observer()).define(["md"], _26);
  main.variable(observer()).define(["unique"], _27);
  main.variable(observer()).define(["unique"], _28);
  main.variable(observer()).define(["md"], _29);
  main.variable(observer("uniqueValid")).define("uniqueValid", ["unique"], _uniqueValid);
  main.variable(observer()).define(["md"], _31);
  main.variable(observer()).define(["uniqueValid"], _32);
  main.variable(observer()).define(["md"], _33);
  main.variable(observer("d3")).define("d3", ["require"], _d3);
  return main;
}
