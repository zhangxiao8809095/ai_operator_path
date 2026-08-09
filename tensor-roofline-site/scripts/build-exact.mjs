import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = resolve(
  projectRoot,
  "source/tensor-roofline-explainer.html",
);
const staticPath = resolve(
  projectRoot,
  "dist/static/tensor-roofline-explainer.html",
);
const workerPath = resolve(projectRoot, "dist/server/index.js");

await mkdir(dirname(staticPath), { recursive: true });
await mkdir(dirname(workerPath), { recursive: true });

const html = await readFile(sourcePath);
await copyFile(sourcePath, staticPath);

const encodedHtml = html.toString("base64");
const worker = `const encodedHtml = ${JSON.stringify(encodedHtml)};

function exactHtmlBytes() {
  const binary = atob(encodedHtml);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

export default {
  async fetch(request) {
    const { pathname } = new URL(request.url);
    if (pathname !== "/" && pathname !== "/tensor-roofline-explainer.html") {
      return new Response("Not Found", { status: 404 });
    }

    return new Response(exactHtmlBytes(), {
      headers: {
        "content-type": "text/html; charset=utf-8",
      },
    });
  },
};
`;

await writeFile(workerPath, worker);
