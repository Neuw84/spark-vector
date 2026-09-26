/*
 * Copyright 2025-2026 Angel Conde and the vecruntime contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.attribute.ModuleHashesAttribute;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Drops the {@code ModuleHashes} attribute from a module-info.class. java.base records the hash of
 * every JDK module, and a rebuilt jdk.incubator.vector (see StripModuleResolution) no longer matches
 * its recorded hash; without the attribute the module path resolves and jlink links the runtime.
 */
public final class StripModuleHashes {
  public static void main(String[] args) throws Exception {
    Path p = Path.of(args[0]);
    ClassFile cf = ClassFile.of();
    byte[] out = cf.transformClass(cf.parse(Files.readAllBytes(p)),
        ClassTransform.dropping(e -> e instanceof ModuleHashesAttribute));
    Files.write(p, out);
  }
}
