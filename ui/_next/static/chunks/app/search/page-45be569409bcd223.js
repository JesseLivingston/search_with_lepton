(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[797],{2116:function(e,t,s){Promise.resolve().then(s.bind(s,9513))},6584:function(e,t,s){"use strict";s.d(t,{C:function(){return o}});var a=s(7437),l=s(6882),n=s(8481),r=s(1396),i=s.n(r),c=s(2265);let o=e=>{let{query:t}=e,s=(0,c.useMemo)(()=>(0,n.x0)(),[t]);return(0,a.jsx)(i(),{prefetch:!1,title:t,href:(0,l.T)(t,s),className:"border border-zinc-200/50 text-ellipsis overflow-hidden text-nowrap items-center rounded-lg bg-zinc-100 hover:bg-zinc-200/80 hover:text-zinc-950 px-2 py-1 text-xs font-medium text-zinc-600",children:t})}},6498:function(e,t,s){"use strict";s.d(t,{o:function(){return o}});var a=s(7437),l=s(6882),n=s(8291),r=s(8481),i=s(4033),c=s(2265);let o=()=>{let[e,t]=(0,c.useState)(""),s=(0,i.useRouter)();return(0,a.jsx)("form",{onSubmit:a=>{a.preventDefault(),e&&(t(""),s.push((0,l.T)(encodeURIComponent(e),(0,r.x0)())))},children:(0,a.jsxs)("label",{className:"relative bg-white flex items-center justify-center border ring-8 ring-zinc-300/20 py-2 px-2 rounded-lg gap-2 focus-within:border-zinc-300",htmlFor:"search-bar",children:[(0,a.jsx)("input",{id:"search-bar",value:e,onChange:e=>t(e.target.value),autoFocus:!0,placeholder:"Ask Lepton AI anything ...",className:"px-2 pr-6 w-full rounded-md flex-1 outline-none bg-white"}),(0,a.jsx)("button",{type:"submit",className:"w-auto py-1 px-2 bg-black border-black text-white fill-white active:scale-95 border overflow-hidden relative rounded-xl",children:(0,a.jsx)(n.Z,{size:16})})]})})}},9513:function(e,t,s){"use strict";s.r(t),s.d(t,{default:function(){return F}});var a=s(7437),l=s(2265),n=s(299),r=s(7042),i=s(4769);function c(){for(var e=arguments.length,t=Array(e),s=0;s<e;s++)t[s]=arguments[s];return(0,i.m6)((0,r.W)(t))}let o=n.fC,d=n.xz,x=l.forwardRef((e,t)=>{let{className:s,align:l="center",sideOffset:r=4,...i}=e;return(0,a.jsx)(n.h_,{children:(0,a.jsx)(n.VY,{ref:t,align:l,sideOffset:r,className:c("z-50 w-72 rounded-md border bg-popover p-4 text-popover-foreground shadow-md outline-none data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",s),...i})})});function m(e){let{className:t,...s}=e;return(0,a.jsx)("div",{className:c("animate-pulse rounded-md bg-muted",t),...s})}x.displayName=n.VY.displayName;let u=e=>{let{title:t,content:s}=e;return(0,a.jsxs)("div",{className:"flex flex-col gap-4 w-full",children:[(0,a.jsx)("div",{className:"flex gap-2 text-blue-500",children:t}),s]})};var h=s(7154),f=s(3881);let p=e=>{let{markdown:t,sources:s}=e;return(0,a.jsx)(u,{title:(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(h.Z,{})," Answer"]}),content:t?(0,a.jsx)("div",{className:"prose prose-sm max-w-full",children:(0,a.jsx)(f.U,{components:{a:e=>{var t,l,n,r;let{node:i,...c}=e;if(!c.href)return(0,a.jsx)(a.Fragment,{});let m=s[+c.href-1];return m?(0,a.jsx)("span",{className:"inline-block w-4",children:(0,a.jsxs)(o,{children:[(0,a.jsx)(d,{asChild:!0,children:(0,a.jsx)("span",{title:m.name,className:"inline-block cursor-pointer transform scale-[60%] no-underline font-medium bg-zinc-300 hover:bg-zinc-400 w-6 text-center h-6 rounded-full origin-top-left",children:c.href})}),(0,a.jsxs)(x,{align:"start",className:"max-w-screen-md flex flex-col gap-2 bg-white shadow-transparent ring-zinc-50 ring-4 text-xs",children:[(0,a.jsx)("div",{className:"text-ellipsis overflow-hidden whitespace-nowrap font-medium",children:m.name}),(0,a.jsxs)("div",{className:"flex gap-4",children:[(null===(t=m.primaryImageOfPage)||void 0===t?void 0:t.thumbnailUrl)&&(0,a.jsx)("div",{className:"flex-none",children:(0,a.jsx)("img",{className:"rounded h-16 w-16",width:null===(l=m.primaryImageOfPage)||void 0===l?void 0:l.width,height:null===(n=m.primaryImageOfPage)||void 0===n?void 0:n.height,src:null===(r=m.primaryImageOfPage)||void 0===r?void 0:r.thumbnailUrl})}),(0,a.jsx)("div",{className:"flex-1",children:(0,a.jsx)("div",{className:"line-clamp-4 text-zinc-500 break-words",children:m.snippet})})]}),(0,a.jsxs)("div",{className:"flex gap-2 items-center",children:[(0,a.jsx)("div",{className:"flex-1 overflow-hidden",children:(0,a.jsx)("div",{className:"text-ellipsis text-blue-500 overflow-hidden whitespace-nowrap",children:(0,a.jsx)("a",{title:m.name,href:m.url,target:"_blank",children:m.url})})}),(0,a.jsx)("div",{className:"flex-none flex items-center relative",children:(0,a.jsx)("img",{className:"h-3 w-3",alt:m.url,src:"https://www.google.com/s2/favicons?domain=".concat(m.url,"&sz=",16)})})]})]})]})}):(0,a.jsx)(a.Fragment,{})}},children:t})}):(0,a.jsxs)("div",{className:"flex flex-col gap-2",children:[(0,a.jsx)(m,{className:"max-w-sm h-4 bg-zinc-200"}),(0,a.jsx)(m,{className:"max-w-lg h-4 bg-zinc-200"}),(0,a.jsx)(m,{className:"max-w-2xl h-4 bg-zinc-200"}),(0,a.jsx)(m,{className:"max-w-lg h-4 bg-zinc-200"}),(0,a.jsx)(m,{className:"max-w-xl h-4 bg-zinc-200"})]})})};var g=s(6584),b=s(3598);let j=e=>{let{relates:t}=e;return(0,a.jsx)(u,{title:(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(b.Z,{})," Related"]}),content:(0,a.jsx)("div",{className:"flex gap-2 flex-col",children:null!==t?t.length>0?t.map(e=>{let{question:t}=e;return(0,a.jsx)(g.C,{query:t},t)}):(0,a.jsx)("div",{className:"text-sm",children:"No related questions."}):(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(m,{className:"w-full h-5 bg-zinc-200/80"}),(0,a.jsx)(m,{className:"w-full h-5 bg-zinc-200/80"}),(0,a.jsx)(m,{className:"w-full h-5 bg-zinc-200/80"})]})})})};var w=s(8468);let v=e=>{let{source:t,index:s}=e,{id:l,name:n,url:r}=t,i=new URL(r).hostname;return(0,a.jsxs)("div",{className:"relative text-xs py-3 px-3 bg-zinc-100 hover:bg-zinc-200 rounded-lg flex flex-col gap-2",children:[(0,a.jsx)("a",{href:r,target:"_blank",className:"absolute inset-0"}),(0,a.jsx)("div",{className:"font-medium text-zinc-950 text-ellipsis overflow-hidden whitespace-nowrap break-words",children:n}),(0,a.jsxs)("div",{className:"flex gap-2 items-center",children:[(0,a.jsx)("div",{className:"flex-1 overflow-hidden",children:(0,a.jsxs)("div",{className:"text-ellipsis whitespace-nowrap break-all text-zinc-400 overflow-hidden w-full",children:[s+1," - ",i]})}),(0,a.jsx)("div",{className:"flex-none flex items-center",children:(0,a.jsx)("img",{className:"h-3 w-3",alt:i,src:"https://www.google.com/s2/favicons?domain=".concat(i,"&sz=",16)})})]})]},l)},N=e=>{let{sources:t}=e;return(0,a.jsx)(u,{title:(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(w.Z,{})," Sources"]}),content:(0,a.jsx)("div",{className:"grid grid-cols-2 sm:grid-cols-4 gap-2",children:t.length>0?t.map((e,t)=>(0,a.jsx)(v,{index:t,source:e},e.id)):(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(m,{className:"max-w-sm h-16 bg-zinc-200/80"}),(0,a.jsx)(m,{className:"max-w-sm h-16 bg-zinc-200/80"}),(0,a.jsx)(m,{className:"max-w-sm h-16 bg-zinc-200/80"}),(0,a.jsx)(m,{className:"max-w-sm h-16 bg-zinc-200/80"})]})})})};async function y(e,t,s,a){let{done:l,value:n}=await e.read();if(l){a&&a(),t.close();return}return s&&s(n),t.enqueue(n),y(e,t,s,a)}let z=(e,t,s)=>{let a=e.body.getReader();return new ReadableStream({start:e=>y(a,e,t,s)})},k="__LLM_RESPONSE__",_="__RELATED_QUESTIONS__",C=async(e,t,s,a,l,n,r)=>{let i=new TextDecoder,c=new Uint8Array,o="",d=!1,x=await fetch("/query",{method:"POST",headers:{"Content-Type":"application/json",Accept:"*./*"},signal:e.signal,body:JSON.stringify({query:t,search_uuid:s})});if(200!==x.status){null==r||r(x.status);return}let m=e=>{l(e.replace(/\[\[([cC])itation/g,"[citation").replace(/[cC]itation:(\d+)]]/g,"citation:$1]").replace(/\[\[([cC]itation:\d+)]](?!])/g,"[$1]").replace(/\[[cC]itation:(\d+)]/g,"[citation]($1)"))};z(x,e=>{if(c=new Uint8Array([...c,...e]),(o=i.decode(c,{stream:!0})).includes(k)){let[e,t]=o.split(k);if(!d)try{a(JSON.parse(e))}catch(e){a([])}if(d=!0,t.includes(_)){let[e]=t.split(_);m(e)}else m(t)}},()=>{let[e,t]=o.split(_);try{n(JSON.parse(t))}catch(e){n([])}})};var S=s(9208);let R=e=>{let{query:t,rid:s}=e,[n,r]=(0,l.useState)([]),[i,c]=(0,l.useState)(""),[o,d]=(0,l.useState)(null),[x,m]=(0,l.useState)(null);return(0,l.useEffect)(()=>{let e=new AbortController;return C(e,t,s,r,c,d,m),()=>{e.abort()}},[t]),(0,a.jsxs)("div",{className:"flex flex-col gap-8",children:[(0,a.jsx)(p,{markdown:i,sources:n}),(0,a.jsx)(N,{sources:n}),(0,a.jsx)(j,{relates:o}),x&&(0,a.jsx)("div",{className:"absolute inset-4 flex items-center justify-center bg-white/40 backdrop-blur-sm",children:(0,a.jsxs)("div",{className:"p-4 bg-white shadow-2xl rounded text-blue-500 font-medium flex gap-4",children:[(0,a.jsx)(S.Z,{}),429===x?"Sorry, you have made too many requests recently, try again later.":"Sorry, we might be overloaded, try again later."]})})]})};var O=s(6498),U=s(6882),I=s(3116),q=s(8481),A=s(4033);let E=e=>{let{query:t}=e,s=(0,A.useRouter)();return(0,a.jsxs)("div",{className:"flex items-center pb-4 mb-6 border-b gap-4",children:[(0,a.jsx)("div",{className:"flex-1 text-lg sm:text-xl text-black text-ellipsis overflow-hidden whitespace-nowrap",title:t,children:t}),(0,a.jsx)("div",{className:"flex-none",children:(0,a.jsxs)("button",{onClick:()=>{s.push((0,U.T)(encodeURIComponent(t),(0,q.x0)()))},type:"button",className:"rounded flex gap-2 items-center bg-transparent px-2 py-1 text-xs font-semibold text-blue-500 hover:bg-zinc-100",children:[(0,a.jsx)(I.Z,{size:12}),"Rewrite"]})})]})};function F(){let e=(0,A.useSearchParams)(),t=decodeURIComponent(e.get("q")||""),s=decodeURIComponent(e.get("rid")||"");return(0,a.jsx)("div",{className:"absolute inset-0 bg-[url('/ui/bg.svg')]",children:(0,a.jsxs)("div",{className:"mx-auto max-w-3xl absolute inset-4 md:inset-8 bg-white",children:[(0,a.jsx)("div",{className:"h-20 pointer-events-none rounded-t-2xl w-full backdrop-filter absolute top-0 bg-gradient-to-t from-transparent to-white [mask-image:linear-gradient(to_bottom,white,transparent)]"}),(0,a.jsxs)("div",{className:"px-4 md:px-8 pt-6 pb-24 rounded-2xl ring-8 ring-zinc-300/20 border border-zinc-200 h-full overflow-auto",children:[(0,a.jsx)(E,{query:t}),(0,a.jsx)(R,{query:t,rid:s},s)]}),(0,a.jsx)("div",{className:"h-80 pointer-events-none w-full rounded-b-2xl backdrop-filter absolute bottom-0 bg-gradient-to-b from-transparent to-white [mask-image:linear-gradient(to_top,white,transparent)]"}),(0,a.jsx)("div",{className:"absolute z-10 flex items-center justify-center bottom-6 px-4 md:px-8 w-full",children:(0,a.jsx)("div",{className:"w-full",children:(0,a.jsx)(O.o,{})})})]})})}},6882:function(e,t,s){"use strict";s.d(t,{T:function(){return a}});let a=(e,t)=>"".concat("/search.html","?q=").concat(encodeURIComponent(e),"&rid=").concat(t)}},function(e){e.O(0,[899,514,971,938,744],function(){return e(e.s=2116)}),_N_E=e.O()}]);