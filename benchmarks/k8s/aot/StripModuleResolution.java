import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.attribute.ModuleResolutionAttribute;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Drops the {@code ModuleResolution} attribute from a module-info.class, so the module is neither
 * "do not resolve by default" nor "incubating". Used on jdk.incubator.vector when the image's runtime
 * is linked (#416): the JDK's AOT cache archives the boot layer only when no incubator module is in
 * the configuration, and the Vector API is one until it leaves incubation.
 */
public final class StripModuleResolution {
  public static void main(String[] args) throws Exception {
    Path p = Path.of(args[0]);
    ClassFile cf = ClassFile.of();
    byte[] out = cf.transformClass(cf.parse(Files.readAllBytes(p)),
        ClassTransform.dropping(e -> e instanceof ModuleResolutionAttribute));
    Files.write(p, out);
  }
}
